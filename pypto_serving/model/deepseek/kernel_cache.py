# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Compiled-program cache keys for DeepSeek V4."""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any, Sequence

from pypto_serving.model.common.kernel_cache import (
    UNKNOWN,
    KernelCache as KernelCache,
    argument_fingerprint,
    pypto_version,
    source_fingerprint,
)


@functools.lru_cache(maxsize=None)
def compute_code_fingerprint(pypto_root: str | None) -> str:
    """Fingerprint the DeepSeek kernel sources and serving compile wrapper."""
    try:
        from pypto_serving.model.deepseek.npu_executor import (  # noqa: PLC0415
            _find_pypto_lib_deepseek_v4_dir,
        )

        kernel_dir = _find_pypto_lib_deepseek_v4_dir(pypto_root)
        sources = [
            (path.relative_to(kernel_dir).as_posix(), path)
            for path in kernel_dir.rglob("*.py")
        ]
        executor_source = Path(__file__).resolve().parent / "npu_executor.py"
        sources.append(("serving/npu_executor.py", executor_source))
        return source_fingerprint(sources)
    except Exception:  # noqa: BLE001 - fingerprint failure safely disables reuse
        return f"{pypto_version()}+{UNKNOWN}"


def compute_params_fingerprint(
    name: str,
    args: Sequence[Any],
    *,
    platform: str,
    block_dim: int | None,
) -> str:
    """Fingerprint all tensor signatures and scalar specializations."""
    return argument_fingerprint(
        name,
        args,
        platform=platform,
        block_dim=block_dim,
    )
