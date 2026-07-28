# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Shared on-disk cache helpers for compiled PyPTO programs."""

from __future__ import annotations

import ast
import hashlib
import logging
import os
import shutil
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

FINGERPRINT_FILE = "fingerprint.txt"
META_FILE = "distributed_meta.json"
UNKNOWN = "unknown"


def canonical_source(src: str) -> bytes:
    """Return a formatting-insensitive canonical form of Python source."""
    try:
        return ast.dump(ast.parse(src), include_attributes=False).encode()
    except SyntaxError:
        return src.encode()


def pypto_version() -> str:
    """Return the installed PyPTO version or an unusable cache sentinel."""
    try:
        import pypto  # noqa: PLC0415

        return str(getattr(pypto, "__version__", UNKNOWN))
    except Exception:  # noqa: BLE001 - version probes must never be fatal
        return UNKNOWN


def source_fingerprint(sources: Iterable[tuple[str, Path]]) -> str:
    """Fingerprint named Python sources together with the PyPTO version."""
    version = pypto_version()
    digest = hashlib.sha256()
    for name, path in sorted(sources):
        digest.update(name.encode())
        digest.update(canonical_source(path.read_text(encoding="utf-8", errors="surrogateescape")))
    return f"{version}+{digest.hexdigest()[:16]}"


def argument_fingerprint(
    name: str,
    args: Sequence[Any],
    *,
    platform: str,
    block_dim: int | None,
) -> str:
    """Fingerprint tensor signatures and scalar values that specialize a program."""
    digest = hashlib.sha256()
    digest.update(name.encode())
    digest.update(platform.encode())
    digest.update(str(block_dim).encode())
    for arg in args:
        shape = getattr(arg, "shape", None)
        dtype = getattr(arg, "dtype", None)
        if shape is not None and dtype is not None:
            descriptor = ("tensor", tuple(shape), str(dtype))
        else:
            value = getattr(arg, "value", arg)
            descriptor = (
                "scalar",
                f"{type(arg).__module__}.{type(arg).__qualname__}",
                repr(value),
            )
        digest.update(repr(descriptor).encode())
    return digest.hexdigest()[:16]


class KernelCache:
    """Load and store compiled programs under one cache slot per program name."""

    def __init__(self, cache_dir: str, code_fingerprint: str) -> None:
        self._cache_dir = Path(cache_dir)
        self._code_fingerprint = code_fingerprint

    def _marker(self, params_fingerprint: str) -> str:
        return f"{self._code_fingerprint}|{params_fingerprint}"

    def _usable(self, params_fingerprint: str) -> bool:
        return UNKNOWN not in self._marker(params_fingerprint)

    def load(
        self,
        name: str,
        params_fingerprint: str,
        *,
        platform: str,
        distributed_config: Any,
    ) -> object | None:
        """Reload a compiled program, returning ``None`` on miss or failure."""
        if not self._usable(params_fingerprint):
            logger.info("[kernel-cache] MISS: %s fingerprint unavailable; compiling", name)
            return None
        try:
            from pypto.ir.distributed_compiled_program import DistributedCompiledProgram  # noqa: PLC0415

            slot = self._cache_dir / name
            if not (slot / META_FILE).exists():
                logger.info("[kernel-cache] MISS: %s not cached under %s; compiling", name, slot)
                return None
            marker_path = slot / FINGERPRINT_FILE
            cached = marker_path.read_text().strip() if marker_path.exists() else None
            current = self._marker(params_fingerprint)
            if cached != current:
                logger.info(
                    "[kernel-cache] STALE: %s cached %r != current %r; recompiling",
                    name,
                    cached,
                    current,
                )
                return None
            compiled = DistributedCompiledProgram.from_dir(
                str(slot),
                platform=platform,
                distributed_config=distributed_config,
            )
            logger.info("[kernel-cache] HIT: reused %s from %s", name, slot)
            return compiled
        except Exception as exc:  # noqa: BLE001 - reuse must never be fatal
            logger.warning(
                "[kernel-cache] reuse of %s failed (%s: %s); recompiling",
                name,
                type(exc).__name__,
                exc,
            )
            return None

    def store(self, name: str, compiled: Any, params_fingerprint: str) -> None:
        """Atomically persist a freshly assembled program into its cache slot."""
        if not self._usable(params_fingerprint):
            return
        try:
            slot = self._cache_dir / name
            src = Path(str(compiled.output_dir))
            if src.resolve() == slot.resolve():
                return
            if not src.exists():
                logger.warning("[kernel-cache] build dir missing for %s; not cached", name)
                return
            slot.parent.mkdir(parents=True, exist_ok=True)
            tmp = slot.with_name(f"{name}.tmp.{os.getpid()}")
            if tmp.exists():
                shutil.rmtree(tmp)
            shutil.copytree(src, tmp)
            (tmp / FINGERPRINT_FILE).write_text(self._marker(params_fingerprint))
            if slot.exists():
                shutil.rmtree(slot)
            os.replace(tmp, slot)
            logger.info("[kernel-cache] STORED: %s (+ device binaries) -> %s", name, slot)
        except Exception as exc:  # noqa: BLE001 - caching must never be fatal
            logger.warning(
                "[kernel-cache] failed to store %s (%s: %s)",
                name,
                type(exc).__name__,
                exc,
            )
