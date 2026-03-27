# Git push (you must run this locally)

## Current state

- Branch: **`dev`**
- **2 commits** ahead of `origin/dev` (not pushed from this environment):
  1. `1ed773f` — checkpoint ignores (`*.pth`, `*.ckpt`, `*.safetensors`)
  2. `9e5cb29` — docs, scripts, ML-SUPERB recipe updates (no `logs/`, no `third_party/wavjepa/`)

Working tree: **clean** (nothing uncommitted).

## Push (after you authenticate as the repo owner)

```bash
cd /path/to/snlp
git fetch origin
git push origin dev
```

If GitHub returns **403** / *Permission denied* (wrong account, e.g. `clmrie` vs `janisaiad`):

1. **SSH (recommended):** set `remote` to SSH and use the right key:
   ```bash
   git remote set-url origin git@github.com:janisaiad/snlp.git
   ssh -T git@github.com
   git push origin dev
   ```
2. **HTTPS + PAT:** use a [Personal Access Token](https://github.com/settings/tokens) as password when prompted, or:
   ```bash
   gh auth login
   git push origin dev
   ```

## If you already pushed the old bad commit

Only if others pulled `7019c9f`-style history:

```bash
git push --force-with-lease origin dev
```

(Only needed if that commit existed on `origin`; after a normal fast-forward push, **do not** force-push.)

## Large files policy (already in `.gitignore`)

- `logs/`, `third_party/wavjepa/`, checkpoints `*.pth` / `*.ckpt` / `*.safetensors`, `*.pt`
- Do **not** `git add` full `data/ml_superb` audio corpora — keep transcripts only if small, or ignore.
