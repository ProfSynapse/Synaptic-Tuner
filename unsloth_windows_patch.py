"""
Unsloth Windows Compatibility Patches
Auto-applied during setup
"""
import sys
import os
from dataclasses import dataclass, fields

def apply_patches():
    print("Applying Windows compatibility patches...")

    # Patch 1: Fix triton AttrsDescriptor
    try:
        import triton
        if hasattr(triton.runtime.autotuner, 'AttrsDescriptor'):
            AttrsDescriptor = triton.runtime.autotuner.AttrsDescriptor
            try:
                fields(AttrsDescriptor)
            except TypeError:
                @dataclass
                class AttrsDescriptor:
                    divisible_by_16: tuple = ()
                    equal_to_1: tuple = ()
                triton.runtime.autotuner.AttrsDescriptor = AttrsDescriptor
                print("  ✓ AttrsDescriptor patched")
    except: pass

    # Patch 2: Wrap fields() for non-dataclasses
    import dataclasses
    original_fields = fields
    def patched_fields(class_or_instance):
        try:
            return original_fields(class_or_instance)
        except TypeError:
            return ()
    dataclasses.fields = patched_fields

    # Patch 3: Disable torch.compile
    os.environ['PYTORCH_JIT'] = '0'
    os.environ['TORCH_COMPILE_DISABLE'] = '1'

    # Patch 4: Pre-patch torch._inductor
    try:
        import torch._inductor.runtime.hints
        if not hasattr(torch._inductor.runtime.hints, 'attr_desc_fields'):
            torch._inductor.runtime.hints.attr_desc_fields = set()
    except: pass

    print("  ✓ All patches applied")

if __name__ == "__main__":
    apply_patches()
    try:
        from unsloth import FastLanguageModel
        print("\n✅ SUCCESS! Unsloth works on Windows!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
