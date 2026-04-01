# WavJEPA-Nat frontend: pretrained JEPA from Hugging Face (labhamlet/wavjepa-nat-base)
# or from a local PyTorch Lightning checkpoint (third_party/wavjepa pretraining).
# Paper: WavJEPA (2509.23238); waveform encoder + ViT, 100 Hz, 768-dim.
from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Optional, Tuple, Union

import humanfriendly
import torch
import torch.nn.functional as F
from typeguard import typechecked

from espnet2.asr.frontend.abs_frontend import AbsFrontend

# WavJEPA-Nat conv downsampling: 5 * 2^4 * 2 = 160 (from conv_layers_spec)
WAVJEPA_STRIDE = 160


def _infer_jepa_size_from_lightning_state(state: dict) -> str:
    """Match third_party/wavjepa/wavjepa/jepa.py width for encoder self-attention."""
    for k, v in state.items():
        if not k.endswith("encoder.layers.0.self_attn.in_proj_weight"):
            continue
        if hasattr(v, "shape") and len(v.shape) >= 2:
            d_model = int(v.shape[1])
            if d_model == 384:
                return "small"
            if d_model == 1024:
                return "large"
            return "base"
    return "base"


def _strip_lightning_state_dict(raw: dict) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for key, value in raw.items():
        if key.startswith("extract_audio._orig_mod"):
            out[key.replace("extract_audio._orig_mod", "extract_audio")] = value
        elif key.startswith("encoder._orig_mod"):
            out[key.replace("encoder._orig_mod", "encoder")] = value
        elif key.startswith("decoder._orig_mod"):
            out[key.replace("decoder._orig_mod", "decoder")] = value
        else:
            out[key] = value
    return out


def _native_normalize(audio: torch.Tensor) -> torch.Tensor:
    mean = audio.mean(dim=(-2, -1), keepdim=True)
    std = audio.std(dim=(-2, -1), keepdim=True)
    return (audio - mean) / (std + 1e-5)


def _calculate_padding_mask(
    pad_frames: int,
    total_frames: int,
    sr: int,
    output_steps: int,
    process_seconds: float,
    device: torch.device,
    batch_size: int,
) -> tuple[torch.Tensor, int]:
    total_frames = int((total_frames / sr) / process_seconds)
    total_output_steps = output_steps * total_frames
    mask = torch.zeros((batch_size, total_output_steps), dtype=torch.bool, device=device)
    output_sr = int(output_steps / process_seconds)
    pad_seconds = pad_frames / sr
    pad_steps = int(pad_seconds * output_sr)
    mask[..., total_output_steps - pad_steps :] = True
    return mask, total_output_steps - pad_steps


def _build_native_jepa_for_inference(
    lightning_ckpt: str,
    in_channels: int,
    process_seconds: float,
    resample_sr: int,
    samples_per_audio: int,
    average_top_k_layers: int,
    compile_modules: bool,
    model_size: Optional[str] = None,
) -> torch.nn.Module:
    from wavjepa.extractors import ConvFeatureExtractor
    from wavjepa.jepa import JEPA
    from wavjepa.types import TransformerEncoderCFG, TransformerLayerCFG

    ckpt = torch.load(lightning_ckpt, map_location="cpu", weights_only=False)
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    if not isinstance(state, dict):
        raise ValueError(f"Unexpected checkpoint format: {lightning_ckpt}")
    stripped = _strip_lightning_state_dict(state)
    size = (model_size or "").strip().lower() or None
    if size not in ("base", "small", "large"):
        size = _infer_jepa_size_from_lightning_state(stripped)

    conv_spec = [(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512, 2, 2)]
    extractor = ConvFeatureExtractor(
        conv_layers_spec=conv_spec,
        in_channels=in_channels,
        depthwise=False,
    )
    model = JEPA(
        feature_extractor=extractor,
        transformer_encoder_cfg=TransformerEncoderCFG.create(),
        transformer_encoder_layers_cfg=TransformerLayerCFG.create(),
        transformer_decoder_cfg=TransformerEncoderCFG.create(),
        transformer_decoder_layers_cfg=TransformerLayerCFG.create(d_model=384),
        resample_sr=resample_sr,
        process_audio_seconds=process_seconds,
        nr_samples_per_audio=samples_per_audio,
        compile_modules=compile_modules,
        average_top_k_layers=average_top_k_layers,
        size=size,
    )
    model.load_state_dict(stripped, strict=False)
    model.eval()
    return model


class WavJEPAFrontend(AbsFrontend):
    """Pretrained WavJEPA-Nat from Hugging Face or from a local Lightning checkpoint."""

    @typechecked
    def __init__(
        self,
        fs: Union[int, str] = 16000,
        frontend_conf: Optional[dict] = None,
        download_dir: Optional[str] = None,
        model_name: str = "labhamlet/wavjepa-nat-base",
        lightning_checkpoint_path: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        pretrain_in_channels: int = 1,
        pretrain_process_seconds: float = 2.01,
        pretrain_samples_per_audio: int = 8,
        pretrain_average_top_k_layers: int = 12,
        pretrain_compile_modules: bool = False,
        pretrain_model_size: Optional[str] = None,
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

        ckpt_path = (
            lightning_checkpoint_path
            or checkpoint_path
            or frontend_conf.get("lightning_checkpoint_path")
            or frontend_conf.get("checkpoint_path")
        )
        self.native_jepa: Optional[torch.nn.Module] = None
        self.extractor = None
        self.encoder = None

        if ckpt_path and Path(str(ckpt_path)).is_file():
            inch = int(frontend_conf.get("pretrain_in_channels", pretrain_in_channels))
            proc_sec = float(frontend_conf.get("pretrain_process_seconds", pretrain_process_seconds))
            spa = int(frontend_conf.get("pretrain_samples_per_audio", pretrain_samples_per_audio))
            atk = int(
                frontend_conf.get("pretrain_average_top_k_layers", pretrain_average_top_k_layers)
            )
            use_compile = bool(
                frontend_conf.get("pretrain_compile_modules", pretrain_compile_modules)
            )
            p_size = frontend_conf.get("pretrain_model_size", pretrain_model_size)
            self.native_jepa = _build_native_jepa_for_inference(
                str(ckpt_path),
                in_channels=inch,
                process_seconds=proc_sec,
                resample_sr=fs,
                samples_per_audio=spa,
                average_top_k_layers=atk,
                compile_modules=use_compile,
                model_size=str(p_size) if p_size else None,
            )
            self.pretrained_params = None
            self.frontend_type = "wavjepa_native_ckpt"
            logging.info("WavJEPA: loaded native Lightning checkpoint from %s", ckpt_path)
            return

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
        if self.native_jepa is not None:
            return int(self.native_jepa.encoder_embedding_dim)
        return 768

    def _forward_native(
        self, input: torch.Tensor, input_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        from einops import rearrange, repeat

        model = self.native_jepa
        assert model is not None
        device = input.device
        dtype = torch.float32
        B = input.size(0)
        unit_frames = int(model.target_length)
        inch = int(model.extract_audio.in_channels)
        output_steps = model.extract_audio.total_patches(unit_frames) // max(inch, 1)
        process_sec = float(unit_frames) / float(self.fs)
        sr = self.fs

        seqs: list[torch.Tensor] = []
        feat_lens: list[int] = []
        max_t = 0
        for b in range(B):
            L = int(input_lengths[b].item())
            wav = input[b : b + 1, :L].unsqueeze(1).to(dtype=dtype)
            cur_frames = L
            pad_frames = unit_frames - (cur_frames % unit_frames)
            if pad_frames == unit_frames:
                pad_frames = 0
            if pad_frames > 0:
                wav = F.pad(wav, (0, pad_frames))
            audio_b = wav.to(device)
            padded_samples = int(audio_b.shape[-1])
            padding_mask, cut_off = _calculate_padding_mask(
                pad_frames,
                padded_samples,
                sr,
                output_steps,
                process_sec,
                device,
                1,
            )
            embeddings: list[torch.Tensor] = []
            mask_idx = 0
            full_len = audio_b.shape[-1]
            n_chunks = full_len // unit_frames
            for i in range(n_chunks):
                chunk = audio_b[..., i * unit_frames : (i + 1) * unit_frames]
                chunk = _native_normalize(chunk)
                mask = padding_mask[:, mask_idx : mask_idx + output_steps]
                with torch.no_grad():
                    mask_e = repeat(mask, "B E -> B (C E)", C=inch)
                    emb = model.get_audio_representation(chunk, mask_e)
                    emb = rearrange(emb, "B (C S) E -> B C S E", C=inch)
                    emb = emb.mean(dim=1)
                embeddings.append(emb)
                mask_idx += output_steps
            x = torch.hstack(embeddings) if embeddings else torch.zeros(
                1, 0, self.output_size(), device=device, dtype=dtype
            )
            x = x[:, :cut_off, :]
            seqs.append(x.squeeze(0))
            feat_lens.append(int(x.size(1)))
            max_t = max(max_t, x.size(1))

        min_frames = 8
        if max_t < min_frames:
            max_t = min_frames
        out = torch.zeros(B, max_t, self.output_size(), device=device, dtype=dtype)
        out_lens = torch.zeros(B, dtype=torch.long, device=device)
        for b in range(B):
            t = seqs[b].size(0)
            sl = min(t, max_t)
            if sl > 0:
                out[b, :sl, :] = seqs[b][:sl, :]
            eff = max(sl, min_frames)
            out_lens[b] = eff
        return out, out_lens

    def forward(
        self, input: torch.Tensor, input_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.native_jepa is not None:
            return self._forward_native(input, input_lengths)

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
        if self.native_jepa is not None:
            logging.info("WavJEPA native checkpoint: skip reload_pretrained_parameters.")
            return
        assert self.encoder is not None and self.pretrained_params is not None
        self.encoder.load_state_dict(self.pretrained_params)
        logging.info("WavJEPA pretrained parameters reloaded.")
