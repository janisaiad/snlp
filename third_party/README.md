# Third-party code

- **wavjepa** – WavJEPA SSL pretraining (JEPA from [labhamlet/wavjepa](https://github.com/labhamlet/wavjepa)).
  - Clone and add as editable dep: from repo root run `./scripts/setup_wavjepa.sh` (clones into `third_party/wavjepa` and runs `uv add --editable third_party/wavjepa`).
  - Then run pretraining (e.g. 10 H100s, AudioSet): `./scripts/run_wavjepa_pretrain.sh --num-gpus 10 --data audioset`.
