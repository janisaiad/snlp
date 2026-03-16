#!/usr/bin/env python3
"""Quick stack check for ML-SUPERB pipeline: imports and CUDA. No data download."""
from __future__ import annotations

import sys


def main() -> int:
    # we check that espnet, s3prl, datasets are importable (same as recipe)
    try:
        import espnet2  # noqa: F401
        print("espnet2:", espnet2.__version__)
    except Exception as e:
        print("espnet2 import failed:", e, file=sys.stderr)
        return 1
    try:
        import s3prl  # noqa: F401
        print("s3prl: ok")
    except Exception as e:
        print("s3prl import failed:", e, file=sys.stderr)
        return 1
    try:
        import datasets  # noqa: F401
        print("datasets: ok")
    except Exception as e:
        print("datasets import failed:", e, file=sys.stderr)
        return 1
    try:
        import torch
        print("torch:", torch.__version__, "cuda:", torch.cuda.is_available())
        if torch.cuda.is_available():
            print("  device:", torch.cuda.get_device_name(0))
    except Exception as e:
        print("torch check failed:", e, file=sys.stderr)
        return 1
    print("Stack check passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
