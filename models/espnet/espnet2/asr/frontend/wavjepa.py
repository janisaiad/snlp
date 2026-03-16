# WavJEPA-Nat frontend: pretrained JEPA from Hugging Face (labhamlet/wavjepa-nat-base).
# Paper: WavJEPA (2509.23238); waveform encoder + ViT, 100 Hz, 768-dim.
from __future__ import annotations

import copy
import logging
from typing import Optional, Tuple, Union

import humanfriendly
import torch
from typeguard import typechecked

from espnet2.asr.frontend.abs_frontend import AbsFrontend

# WavJEPA-Nat conv downsampling: 5 * 2^4 * 2 = 160 (from conv_layers_spec)
WAVJEPA_STRIDE = 160


class WavJEPAFrontend(AbsFrontend):
    """Pretrained WavJEPA-Nat from Hugging Face (labhamlet/wavjepa-nat-base)."""

    @typechecked
    def __init__(
        self,
        fs: Union[int, str] = 16000,
        frontend_conf: Optional[dict] = None,
        download_dir: Optional[str] = None,
        model_name: str = "labhamlet/wavjepa-nat-base",
    ):
        try:
            from transformers import AutoFeatureExtractor, AutoModel
        except ImportError:
            raise ImportError("Install transformers: uv add transformers")

        super().__init__()
        frontend_conf = frontend_conf or {}
        if isinstance(fs, str):
            fs = int(humanfriendly.parse_size(fs))
        self.fs = fs
        model_name = frontend_conf.get("model_name", model_name)
        cache_dir = frontend_conf.get("download_dir") or download_dir

        self.extractor = AutoFeatureExtractor.from_pretrained(
            model_name, trust_remote_code=True, cache_dir=cache_dir
        )
        self.encoder = AutoModel.from_pretrained(
            model_name, trust_remote_code=True, cache_dir=cache_dir
        )
        if self.extractor.sampling_rate != fs:
            raise ValueError(
                f"WavJEPA-Nat expects 16 kHz; got fs={fs}. Resample or set fs=16000."
            )
        self.encoder.eval()
        self.pretrained_params = copy.deepcopy(self.encoder.state_dict())
        self.frontend_type = "wavjepa"

    def output_size(self) -> int:
        return 768

    def forward(
        self, input: torch.Tensor, input_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # input: (B, L) waveform
        B = input.size(0)
        device = input.device
        # extractor expects list of 1d arrays; mono is duplicated to stereo inside
        waveforms = [input[b, : input_lengths[b]].cpu().numpy() for b in range(B)]
        batch = self.extractor(
            waveforms,
            return_tensors="pt",
            sampling_rate=self.extractor.sampling_rate,
            padding=True,
        )
        input_values = batch["input_values"].to(device)
        with torch.no_grad():
            out = self.encoder(input_values)
        if isinstance(out, tuple):
            out = out[0]
        # (B, 2, T, 768) -> (B, T, 768) by averaging over channels
        if out.dim() == 4:
            out = out.mean(dim=1)
        if out.dim() == 2:
            out = out.unsqueeze(0)
        out = out.clone()  # avoid inference tensor so specaug can do in-place
        # ~200 frames per 2.01s at 16kHz -> stride ~161
        feats_lens = (input_lengths + 160) // 161
        feats_lens = feats_lens.clamp(max=out.size(1), min=1)
        # pad to at least 8 frames so transformer subsampler (needs >7) does not error
        min_frames = 8
        if out.size(1) < min_frames:
            out = torch.nn.functional.pad(
                out, (0, 0, 0, min_frames - out.size(1)), value=0.0
            )
            feats_lens = feats_lens.clamp(min=min_frames)
        return out, feats_lens

    def reload_pretrained_parameters(self) -> None:
        self.encoder.load_state_dict(self.pretrained_params)
        logging.info("WavJEPA pretrained parameters reloaded.")
