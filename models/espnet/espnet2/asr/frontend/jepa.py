# JEPA (Joint Embedding Predictive Architecture) frontend for ASR.
# Uses mel-spectrogram + encoder; optional: Sony audio-representations ViT
# (https://github.com/SonyCSLParis/audio-representations). When jepa_repo_path
# is set and the repo is available, uses their ViTEncoder; otherwise uses a
# minimal built-in patch encoder (no external JEPA deps).
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from typeguard import typechecked

from espnet2.asr.frontend.abs_frontend import AbsFrontend

# mel: 80 bins, 16 kHz, 25 ms window, 10 ms hop (same as Sony audio-representations)
JEPA_MEL_HOP = 160
JEPA_MEL_NFFT = 400
JEPA_MEL_NMELS = 80
JEPA_MEL_FMIN = 50
JEPA_MEL_FMAX = 8000
JEPA_PATCH_SIZE = 16
JEPA_EMBED_DIM = 768


def _make_mel(sample_rate: int = 16000) -> nn.Module:
    try:
        import torchaudio.transforms as T
    except ImportError:
        raise ImportError("torchaudio is required for JEPA frontend (mel).")
    return T.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=JEPA_MEL_NFFT,
        win_length=JEPA_MEL_NFFT,
        hop_length=JEPA_MEL_HOP,
        n_mels=JEPA_MEL_NMELS,
        f_min=JEPA_MEL_FMIN,
        f_max=JEPA_MEL_FMAX,
        power=2.0,
    )


class MinimalJEPAEncoder(nn.Module):
    """Minimal mel-patch encoder: Conv2d patch embed -> (B, n_patches, embed_dim)."""

    def __init__(self, embed_dim: int = JEPA_EMBED_DIM, patch_size: int = JEPA_PATCH_SIZE):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        # (B, 1, 80, T) -> (B, embed_dim, 80//ps, T//ps)
        self.patch_embed = nn.Conv2d(
            1, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        self.n_freq_patches = JEPA_MEL_NMELS // patch_size  # 5

    def forward(
        self, mel: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # mel: (B, 1, 80, T)
        x = self.patch_embed(mel)  # (B, D, 5, T//16)
        B, D, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, H*W, D)
        return x


class JEPAFrontend(AbsFrontend):
    """JEPA-style frontend: wav -> log-mel -> encoder -> (feats, feats_lens)."""

    @typechecked
    def __init__(
        self,
        fs: Union[int, str] = 16000,
        frontend_conf: Optional[dict] = None,
        download_dir: Optional[str] = None,
        jepa_repo_path: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        embed_dim: int = JEPA_EMBED_DIM,
    ):
        super().__init__()
        frontend_conf = frontend_conf or {}
        if isinstance(fs, str):
            import humanfriendly
            fs = int(humanfriendly.parse_size(fs))
        self.fs = fs
        self.embed_dim = embed_dim
        self.mel = _make_mel(fs)
        self.log_eps = 1e-5

        # optional: load Sony ViT from audio-representations repo
        self._encoder: Optional[nn.Module] = None
        self._encoder_is_sony = False
        repo_path = jepa_repo_path or frontend_conf.get("jepa_repo_path")
        ckpt_path = checkpoint_path or frontend_conf.get("checkpoint_path")

        if repo_path and Path(repo_path).exists():
            try:
                self._encoder = self._load_sony_encoder(repo_path, ckpt_path)
                self._encoder_is_sony = True
            except Exception as e:
                logging.warning(
                    "JEPA: could not load Sony encoder from %s: %s; using minimal encoder.",
                    repo_path, e,
                )

        if self._encoder is None:
            self._encoder = MinimalJEPAEncoder(embed_dim=embed_dim)

        self.pretrained_params = None
        if ckpt_path and Path(ckpt_path).exists() and not self._encoder_is_sony:
            try:
                state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
                if isinstance(state, dict) and "state_dict" in state:
                    state = state["state_dict"]
                self._encoder.load_state_dict(state, strict=False)
                self.pretrained_params = {k: v.clone() for k, v in self._encoder.state_dict().items()}
            except Exception as e:
                logging.warning("JEPA: could not load checkpoint %s: %s", ckpt_path, e)

        self._encoder.eval()
        self.hop_length = JEPA_MEL_HOP
        self.frontend_type = "jepa"

    def _load_sony_encoder(self, repo_path: str, checkpoint_path: Optional[str] = None) -> nn.Module:
        import sys
        repo_root = Path(repo_path).resolve()
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
        try:
            from src.models.components.vision_transformer import ViTEncoder
        except ImportError as e:
            raise ImportError(
                "Sony JEPA encoder requires audio-representations repo and deps (e.g. timm). "
                "Clone https://github.com/SonyCSLParis/audio-representations and install: "
                "pip install timm"
            ) from e
        # Sony default: img_size [80, 208], patch 16, embed_dim 768
        encoder = ViTEncoder(
            img_size=[80, 208],
            in_chans=1,
            patch_size=16,
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4.0,
            flash_attn=getattr(torch.backends.cuda, "flash_sdp_enabled", lambda: False)(),
        )
        if checkpoint_path and Path(checkpoint_path).exists():
            ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            if isinstance(ckpt, dict) and "state_dict" in ckpt:
                ckpt = ckpt["state_dict"]
            enc_state = {k.replace("encoder.", ""): v for k, v in ckpt.items() if k.startswith("encoder.")}
            if enc_state:
                encoder.load_state_dict(enc_state, strict=False)
        return encoder

    def output_size(self) -> int:
        return self.embed_dim

    def _wav_to_logmel(self, input: torch.Tensor) -> torch.Tensor:
        # input: (B, 1, L), 16 kHz; output (B, 80, T_mel)
        mel = self.mel(input.squeeze(1))
        mel = (mel + self.log_eps).log()
        return mel

    def forward(
        self, input: torch.Tensor, input_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # input: (B, 1, L)
        mel = self._wav_to_logmel(input).unsqueeze(1)  # (B, 1, 80, T)

        if self._encoder_is_sony:
            # Sony ViT expects fixed (80, 208); chunk and concat
            feats_list, feats_lens_list = [], []
            win = 208
            hop = 104
            for b in range(mel.shape[0]):
                T_m = mel.shape[3]
                out_frames = []
                for start in range(0, max(1, T_m - win + 1), hop):
                    end = min(start + win, T_m)
                    patch = mel[b : b + 1, :, :, start:end]
                    if patch.shape[3] < win:
                        patch = nn.functional.pad(patch, (0, win - patch.shape[3]))
                    with torch.no_grad():
                        out = self._encoder(patch, mask=None)  # (1, 65, 768)
                    out_frames.append(out.squeeze(0))
                if not out_frames:
                    pad = mel[b : b + 1, :, :, :win]
                    if pad.shape[3] < win:
                        pad = nn.functional.pad(pad, (0, win - pad.shape[3]))
                    with torch.no_grad():
                        out = self._encoder(pad, mask=None)
                    out_frames = [out.squeeze(0)]
                feats_b = torch.cat(out_frames, dim=0)  # (T_out, 768)
                feats_list.append(feats_b)
                feats_lens_list.append(feats_b.shape[0])
            feats = torch.nn.utils.rnn.pad_sequence(feats_list, batch_first=True)
            feats_lens = torch.tensor(feats_lens_list, device=input.device, dtype=torch.long)
        else:
            # minimal encoder: (B, 1, 80, T) -> (B, n_patches, 768)
            with torch.no_grad():
                feats = self._encoder(mel)  # (B, n_patches, 768)
            patch_freq = JEPA_MEL_NMELS // JEPA_PATCH_SIZE  # 5
            # mel frames from input_lengths: 100 frames per second at 16kHz; patch_time = mel_frames // 16
            mel_frames = (input_lengths.float() * 100.0 / self.fs).long().clamp(1)
            patch_time = mel_frames // JEPA_PATCH_SIZE
            feats_lens = (patch_time * patch_freq).clamp(max=feats.shape[1])

        return feats, feats_lens

    def reload_pretrained_parameters(self) -> None:
        if self.pretrained_params is not None:
            self._encoder.load_state_dict(self.pretrained_params)
            logging.info("JEPA frontend pretrained parameters reloaded.")
