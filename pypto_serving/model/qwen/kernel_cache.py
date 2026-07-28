# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""On-disk cache of compiled Qwen3-14B PyPTO kernels.

Reloading a compiled kernel (IR + device binaries) from disk skips both the JIT
and the ~30s device-binary compile on every launch. Correctness hinges on the
cache key: a kernel binary is specialised by *both*

  * the compile parameters (batch, max_seq, page_size, vocab, head_dim,
    block-table stride, target platform, ...), and
  * the code that fed the compiler (the pypto-lib kernels, the host dispatch
    wrappers in this package, and the pypto compiler itself).

so BOTH must match before a cached slot may be reused. The key is split across
two orthogonal fingerprints:

  * ``params`` -> encoded into the slot marker; a config change is a guaranteed
    MISS, never a stale HIT.
  * ``code``   -> encoded into the slot marker; a kernel/dispatch edit or a
    pypto upgrade is a STALE and forces a recompile + overwrite.

There is one slot per kernel name, overwritten in place, so only the latest
copy is ever kept. All reuse/store is best-effort and never fatal: any failure
falls back to a fresh compile.
"""

from __future__ import annotations

import functools
import hashlib
from pathlib import Path
from typing import Any, Sequence

from pypto_serving.model.common.kernel_cache import (
    FINGERPRINT_FILE as FINGERPRINT_FILE,
    META_FILE as META_FILE,
    UNKNOWN,
    KernelCache as KernelCache,
    canonical_source as canonical_source,
    pypto_version,
    source_fingerprint,
)


@functools.lru_cache(maxsize=None)
def compute_code_fingerprint(pypto_root: str | None) -> str:
    """Fingerprint of everything that feeds codegen for the Qwen3-14B kernels.

    Combines the pypto compiler version with an AST-canonical hash of every
    source that is actually compiled: the pypto-lib Qwen3-14B kernel modules
    *and* the host dispatch wrappers in this package (``qwen3_l3_dispatch.py``),
    which are the functions handed to ``jit_fn.compile``. Hashing the sources
    -- not a submodule SHA -- also catches uncommitted local edits, exactly the
    dev-iteration case the cache must not serve stale.

    Memoised per process: the sources cannot change under a running launch.
    Any probe failure degrades to a marker containing ``UNKNOWN``, which the
    cache treats as a permanent MISS, forcing a safe recompile.
    """
    try:
        # Imported lazily to avoid a module import cycle (npu_executor imports
        # this module at top level) and to keep this module torch-free.
        from pypto_serving.model.qwen.npu_executor import _find_pypto_lib_qwen14b_dir  # noqa: PLC0415

        kernel_dir = _find_pypto_lib_qwen14b_dir(pypto_root)
        dispatch = Path(__file__).resolve().parent / "qwen3_l3_dispatch.py"

        sources: list[tuple[str, Path]] = [
            (path.relative_to(kernel_dir).as_posix(), path)
            for path in kernel_dir.rglob("*.py")
        ]
        if dispatch.is_file():
            sources.append((dispatch.name, dispatch))

        return source_fingerprint(sources)
    except Exception:  # noqa: BLE001 - fingerprint must never be fatal
        return f"{pypto_version()}+{UNKNOWN}"


def compute_params_fingerprint(
    name: str,
    dummy_args: Sequence[Any],
    *,
    platform: str,
) -> str:
    """Fingerprint of everything that specialises this kernel's binary.

    The ``dummy_args`` shape+dtype signature encodes every compile-time
    dimension (batch, max_seq, page_size, vocab, head_dim, block-table stride,
    ...); combined with the kernel name and target platform it uniquely
    identifies the compiled kernel. Two launches differing in any of these
    produce different fingerprints, so a config change can never be served a
    stale binary.
    """
    digest = hashlib.sha256()
    digest.update(name.encode())
    digest.update(platform.encode())
    for arg in dummy_args:
        digest.update(str(tuple(arg.shape)).encode())
        digest.update(str(arg.dtype).encode())
    return digest.hexdigest()[:16]
