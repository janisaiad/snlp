# Layer Importance Package (split for GitHub LFS)

This folder contains a split archive of the checkpoint package:

- `layer_importance_package.tar.part-000`
- `layer_importance_package.tar.part-001`
- `layer_importance_package.tar.part-002`
- `layer_importance_package.tar.part-003`
- `SHA256SUMS.txt`

## Rebuild command (receiver side)

```bash
cat layer_importance_package.tar.part-* > layer_importance_package.tar
sha256sum -c SHA256SUMS.txt
tar -xf layer_importance_package.tar
```

Reconstructed top-level directory:

- `layer_importance_package/models/*.pth`
- `layer_importance_package/notes/MANIFEST.txt`
