# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Unit tests for the compiled-kernel disk cache.

These exercise the cache key / marker logic without an NPU or the real pypto
compiler: fake tensors stand in for the compile-time dummy args, and a fake
``DistributedCompiledProgram`` stands in for the reload path.
"""

import ctypes
import sys
import types

import pytest

from pypto_serving.model.qwen.kernel_cache import (
    FINGERPRINT_FILE,
    META_FILE,
    KernelCache,
    canonical_source,
    compute_params_fingerprint,
)
from pypto_serving.model.deepseek.kernel_cache import (
    compute_params_fingerprint as compute_deepseek_params_fingerprint,
)


class _FakeTensor:
    """Minimal stand-in for a torch tensor: only ``shape`` and ``dtype`` are
    read by ``compute_params_fingerprint``."""

    def __init__(self, shape, dtype="bfloat16"):
        self.shape = tuple(shape)
        self.dtype = dtype


class _FakeCompiled:
    """Stand-in for a compiled program: only ``output_dir`` is read by store."""

    def __init__(self, output_dir):
        self.output_dir = str(output_dir)


def _install_fake_pypto(monkeypatch, from_dir):
    """Inject a fake ``pypto.ir.distributed_compiled_program`` so ``load`` can
    resolve ``DistributedCompiledProgram.from_dir`` without the real compiler."""

    class _FakeProgram:
        pass

    # Set outside the class body: inside it, ``from_dir`` on the RHS would not
    # resolve to the parameter (class-body scoping).
    _FakeProgram.from_dir = staticmethod(from_dir)

    leaf = types.ModuleType("pypto.ir.distributed_compiled_program")
    leaf.DistributedCompiledProgram = _FakeProgram
    monkeypatch.setitem(sys.modules, "pypto", types.ModuleType("pypto"))
    monkeypatch.setitem(sys.modules, "pypto.ir", types.ModuleType("pypto.ir"))
    monkeypatch.setitem(sys.modules, "pypto.ir.distributed_compiled_program", leaf)


@pytest.fixture
def reload_calls(monkeypatch):
    """A fake compiler whose ``from_dir`` records calls and returns a sentinel
    tagged with the slot it was asked to reload."""
    calls = []

    def from_dir(path, *, platform, distributed_config):
        calls.append((path, platform))
        return ("reloaded", path)

    _install_fake_pypto(monkeypatch, from_dir)
    return calls


def _make_build_dir(tmp_path, name="build", payload=b"binary"):
    """A fake kernel build dir: a meta file plus a device binary."""
    build = tmp_path / name
    (build / "cache").mkdir(parents=True)
    (build / META_FILE).write_text("{}")
    (build / "cache" / "k.bin").write_bytes(payload)
    return build


def _cache(tmp_path, code_fingerprint="v1+abcd"):
    return KernelCache(str(tmp_path / "cache"), code_fingerprint)


# --- params fingerprint ----------------------------------------------------

def _pf(name, args, platform="a2a3"):
    return compute_params_fingerprint(name, args, platform=platform)


def test_params_fingerprint_is_stable():
    a = [_FakeTensor((16, 512)), _FakeTensor((512,))]
    b = [_FakeTensor((16, 512)), _FakeTensor((512,))]
    assert _pf("decode_fwd", a) == _pf("decode_fwd", b)


def test_params_fingerprint_tracks_every_distinguishing_dimension():
    args = [_FakeTensor((16, 512), "bfloat16")]
    base = _pf("decode_fwd", args)
    # max_seq (shape) change -> the exact bug the name-only key missed
    assert base != _pf("decode_fwd", [_FakeTensor((16, 2048), "bfloat16")])
    # dtype, platform, and kernel name all distinguish a binary
    assert base != _pf("decode_fwd", [_FakeTensor((16, 512), "float32")])
    assert base != _pf("decode_fwd", args, platform="a2a3sim")
    assert base != _pf("prefill_fwd", args)


def test_deepseek_params_fingerprint_tracks_scalar_values():
    args = [_FakeTensor((8, 512)), ctypes.c_int32(8)]
    base = compute_deepseek_params_fingerprint(
        "deepseek_v4_mtp_decode",
        args,
        platform="a2a3",
        block_dim=None,
    )
    changed = compute_deepseek_params_fingerprint(
        "deepseek_v4_mtp_decode",
        [_FakeTensor((8, 512)), ctypes.c_int32(128)],
        platform="a2a3",
        block_dim=None,
    )
    assert base != changed


# --- AST-canonical source hashing -----------------------------------------

def test_canonical_source_ignores_blank_lines_comments_and_spacing():
    a = "def f(x):\n    return x + 1\n"
    b = "def f(x):\n\n    # a new comment\n    return x  +  1\n\n\n"
    assert canonical_source(a) == canonical_source(b)


def test_canonical_source_detects_logic_change():
    a = "def f(x):\n    return x + 1\n"
    c = "def f(x):\n    return x + 2\n"
    assert canonical_source(a) != canonical_source(c)


def test_canonical_source_falls_back_on_syntax_error():
    broken = "def f(:\n"
    assert canonical_source(broken) == broken.encode()


# --- KernelCache load / store ---------------------------------------------

def test_store_then_load_round_trip(tmp_path, reload_calls):
    cache = _cache(tmp_path)
    build = _make_build_dir(tmp_path)
    cache.store("decode_fwd", _FakeCompiled(build), "pf1")

    slot = tmp_path / "cache" / "decode_fwd"
    assert (slot / META_FILE).exists()
    assert (slot / "cache" / "k.bin").exists()
    assert (slot / FINGERPRINT_FILE).read_text() == "v1+abcd|pf1"

    out = cache.load("decode_fwd", "pf1", platform="a2a3", distributed_config=None)
    assert out == ("reloaded", str(slot))
    assert reload_calls == [(str(slot), "a2a3")]


def test_load_miss_when_params_differ(tmp_path, reload_calls):
    cache = _cache(tmp_path)
    cache.store("decode_fwd", _FakeCompiled(_make_build_dir(tmp_path)), "paramsA")
    # A different runtime config yields a different params fingerprint: this
    # must MISS (recompile), never serve the stale binary.
    assert cache.load("decode_fwd", "paramsB", platform="a2a3", distributed_config=None) is None
    assert reload_calls == []  # from_dir never reached


def test_load_stale_when_code_fingerprint_differs(tmp_path, reload_calls):
    _cache(tmp_path, "v1+abcd").store("decode_fwd", _FakeCompiled(_make_build_dir(tmp_path)), "pf")
    # A kernel-source edit / pypto upgrade changes the code fingerprint.
    newer = _cache(tmp_path, "v2+zzzz")
    assert newer.load("decode_fwd", "pf", platform="a2a3", distributed_config=None) is None
    assert reload_calls == []


def test_load_miss_when_not_cached(tmp_path, reload_calls):
    assert _cache(tmp_path).load("decode_fwd", "pf", platform="a2a3", distributed_config=None) is None


def test_load_corrupt_slot_falls_back(tmp_path, monkeypatch):
    def boom(path, *, platform, distributed_config):
        raise RuntimeError("corrupt artifact")

    _install_fake_pypto(monkeypatch, boom)
    cache = _cache(tmp_path)
    cache.store("decode_fwd", _FakeCompiled(_make_build_dir(tmp_path)), "pf")
    # Marker matches, but from_dir blows up: best-effort reuse returns None.
    assert cache.load("decode_fwd", "pf", platform="a2a3", distributed_config=None) is None


def test_unknown_fingerprint_never_stores_or_hits(tmp_path, reload_calls):
    build = _make_build_dir(tmp_path)
    unusable = _cache(tmp_path, "1.2.3+unknown")  # source hash probe failed
    unusable.store("decode_fwd", _FakeCompiled(build), "pf")
    slot = tmp_path / "cache" / "decode_fwd"
    assert not slot.exists()  # store is a no-op under an unusable fingerprint

    # Even against a good pre-existing slot, an unusable fingerprint must MISS —
    # two "unknown" markers must never compare equal and HIT foreign code.
    _cache(tmp_path, "v1+abcd").store("decode_fwd", _FakeCompiled(build), "pf")
    assert unusable.load("decode_fwd", "pf", platform="a2a3", distributed_config=None) is None
    assert reload_calls == []


def test_store_overwrites_single_slot(tmp_path, reload_calls):
    cache = _cache(tmp_path)
    cache.store("decode_fwd", _FakeCompiled(_make_build_dir(tmp_path, "b1", b"OLD")), "pfOLD")
    cache.store("decode_fwd", _FakeCompiled(_make_build_dir(tmp_path, "b2", b"NEW")), "pfNEW")

    slot = tmp_path / "cache" / "decode_fwd"
    # Only the latest copy survives: one slot, overwritten in place.
    assert (slot / FINGERPRINT_FILE).read_text() == "v1+abcd|pfNEW"
    assert (slot / "cache" / "k.bin").read_bytes() == b"NEW"
    # No half-written temp dirs left behind.
    leftovers = [p.name for p in (tmp_path / "cache").iterdir() if ".tmp." in p.name]
    assert leftovers == []


def test_store_is_noop_when_build_dir_is_the_slot(tmp_path, reload_calls):
    cache = _cache(tmp_path)
    slot = tmp_path / "cache" / "decode_fwd"
    (slot / "cache").mkdir(parents=True)
    (slot / META_FILE).write_text("{}")
    before = sorted(p.name for p in slot.rglob("*"))

    # A kernel reused from cache has output_dir == its slot; storing must not
    # rmtree-then-copy the slot onto itself.
    cache.store("decode_fwd", _FakeCompiled(slot), "pf")
    after = sorted(p.name for p in slot.rglob("*"))
    assert before == after
    assert not (slot / FINGERPRINT_FILE).exists()
