#!/usr/bin/env python3
"""Apply compatibility patches to s3prl in the current venv (Python 3.12+, torchaudio 2.1+).
Run after: uv add s3prl  (or uv sync)
Usage: uv run python scripts/patch_s3prl_for_ssl.py
"""
from __future__ import annotations

import sys
from pathlib import Path


def find_s3prl_root() -> Path | None:
    try:
        import s3prl
        return Path(s3prl.__file__).resolve().parent
    except ImportError:
        return None


def patch_byol_common(root: Path) -> bool:
    p = root / "upstream" / "byol_s" / "byol_a" / "common.py"
    if not p.is_file():
        return False
    text = p.read_text()
    if 'hasattr(torchaudio, "set_audio_backend")' in text:
        return True
    # fresh s3prl has a single line
    old1 = 'torchaudio.set_audio_backend("sox_io")'
    new_block = '''# set_audio_backend was removed in torchaudio 2.1+
if hasattr(torchaudio, "set_audio_backend"):
    torchaudio.set_audio_backend("sox_io")'''
    if old1 in text and new_block not in text:
        text = text.replace(old1, new_block)
        p.write_text(text)
        return True
    return False


def patch_roberta_model(root: Path) -> bool:
    p = root / "upstream" / "roberta" / "roberta_model.py"
    if not p.is_file():
        return False
    text = p.read_text()
    changed = False
    if "encoder: EncDecBaseConfig = EncDecBaseConfig()" in text:
        text = text.replace(
            "encoder: EncDecBaseConfig = EncDecBaseConfig()",
            "encoder: EncDecBaseConfig = field(default_factory=EncDecBaseConfig)",
        )
        changed = True
    if "decoder: DecoderConfig = DecoderConfig()" in text:
        text = text.replace(
            "decoder: DecoderConfig = DecoderConfig()",
            "decoder: DecoderConfig = field(default_factory=DecoderConfig)",
        )
        changed = True
    if "quant_noise: QuantNoiseConfig = QuantNoiseConfig()" in text:
        text = text.replace(
            "quant_noise: QuantNoiseConfig = QuantNoiseConfig()",
            "quant_noise: QuantNoiseConfig = field(default_factory=QuantNoiseConfig)",
        )
        changed = True
    if changed:
        p.write_text(text)
    return True


def patch_mos_prediction_expert(root: Path) -> bool:
    p = root / "upstream" / "mos_prediction" / "expert.py"
    if not p.is_file():
        return False
    text = p.read_text()
    if "apply_effects_tensor = None" in text or ("try:" in text and "sox_effects" in text):
        return True
    old = "from torchaudio.sox_effects import apply_effects_tensor\n\nimport s3prl"
    new = """try:
    from torchaudio.sox_effects import apply_effects_tensor
except (ImportError, ModuleNotFoundError):
    apply_effects_tensor = None  # torchaudio 2.1+ removed sox_effects

import s3prl"""
    if old not in text:
        return False
    p.write_text(text.replace(old, new))
    return True


def main() -> int:
    root = find_s3prl_root()
    if root is None:
        print("s3prl not found in this environment. Run: uv add s3prl", file=sys.stderr)
        return 1
    print(f"Patching s3prl at {root}")
    ok1 = patch_byol_common(root)
    ok2 = patch_roberta_model(root)
    ok3 = patch_mos_prediction_expert(root)
    print(f"  byol_a/common.py: {'patched' if ok1 else 'already patched or missing'}")
    print(f"  roberta/roberta_model.py: {'patched' if ok2 else 'already patched or missing'}")
    print(f"  mos_prediction/expert.py: {'patched' if ok3 else 'already patched or missing'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
