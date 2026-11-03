"""Generic weight-patch capability for MechInterp (experimental, CLI-only).

  spec.py        the declarative ``mechinterp-weight-patch/v1`` schema
  safetensors_io strict pickle-free safetensors header parsing + byte hashing
  checkpoint.py  inspect/bind a LOCAL safetensors checkpoint
  patch.py       validate + apply patch operations (pure, transactional)
  runtime.py     in-memory application to a loaded torch module (revertable)
  export.py      create-only export of an exact modified checkpoint + manifest

Nothing here names a model, tensor, direction or dataset: every selection,
shape, value and restriction comes from the spec and its payload. Importing the
package does not import torch; only patch/runtime/export do.
"""
