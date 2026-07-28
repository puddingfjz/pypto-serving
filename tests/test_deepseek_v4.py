# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch
from pypto.runtime import DeviceTensor

import pypto_serving.cli.main as cli
from pypto_serving.config.types import DecodeBatch, PrefillBatch, RuntimeConfig
from pypto_serving.model import model_loader
from pypto_serving.model import tokenizer as tokenizer_module
from pypto_serving.model.deepseek import npu_executor, weight_loader
from pypto_serving.model.deepseek.npu_runner import (
    DEEPSEEK_V4_LM_HEAD_TP_SIZE,
    DeepSeekV4CacheLayout,
    DeepSeekV4CacheMetadataBuilder,
    DeepSeekV4CompiledKernels,
    DeepSeekV4L3Callable,
    DeepSeekV4ModelRunner,
    accept_mtp_tokens,
    build_deepseek_v4_cache_group_specs,
    build_deepseek_v4_layer_plan,
    deepseek_v4_cache_blocks_for_slots,
)
from pypto_serving.model.deepseek.weight_loader import (
    DeepSeekV4WeightStore,
    deepseek_v4_layer_core_weight_names,
    deepseek_v4_hadamard_idx,
    deepseek_v4_local_expert_ids,
    deepseek_v4_routed_expert_weight_names,
    deepseek_v4_startup_weight_names,
    pack_deepseek_v4_layer_weights,
)
from pypto_serving.model.model_loader import ModelLoader


def test_deepseek_kernel_dir_uses_v4_flash_variant(tmp_path):
    kernel_dir = tmp_path / "models" / "deepseek" / "v4-flash"
    kernel_dir.mkdir(parents=True)

    assert npu_executor._find_pypto_lib_deepseek_v4_dir(str(tmp_path)) == kernel_dir
    assert npu_executor._is_deepseek_v4_module_file(kernel_dir / "decode_fwd.py", kernel_dir)


def test_accept_mtp_tokens_commits_second_main_token_only_on_draft_match():
    accepted = accept_mtp_tokens(
        torch.tensor([[11, 12], [21, 22]], dtype=torch.long),
        torch.tensor([11, 99], dtype=torch.long),
    )

    assert accepted == [[11, 12], [21]]


def test_deepseek_mtp_speculative_batch_feeds_current_then_draft():
    runner, model = _runner_for_prepared_inputs()
    embed = torch.arange(model.config.vocab_size * model.config.hidden_size, dtype=torch.float32).reshape(
        model.config.vocab_size, model.config.hidden_size
    )
    runner._compiled.weight_store = type(
        "Store",
        (),
        {"load_tensor": staticmethod(lambda name: embed if name == "embed.weight" else None)},
    )()
    batch = DecodeBatch(
        request_ids=["req-a"],
        token_ids=torch.tensor([[2]], dtype=torch.long),
        hidden_states=embed[2:3].to(torch.bfloat16),
        seq_lens=torch.tensor([129], dtype=torch.int32),
        allow_device_greedy_sampling=True,
    )

    speculative = runner._main_speculative_batch(model, batch, torch.tensor([5]))

    assert speculative.prev_token_ids.tolist() == [2]
    assert speculative.token_ids.tolist() == [[5]]
    assert speculative.seq_lens.tolist() == [130]
    torch.testing.assert_close(speculative.prev_hidden_states, embed[2:3].to(torch.bfloat16))
    torch.testing.assert_close(speculative.hidden_states, embed[5:6].to(torch.bfloat16))


def test_deepseek_mtp_committed_window_handles_rejection_and_acceptance():
    runner, _model = _runner_for_prepared_inputs()
    inputs = SimpleNamespace(
        request_ids=("req-a",),
        ranks=(0,),
        local_rows=(0,),
        position_ids=torch.tensor(
            [[7, 8, 0, 0, 0, 0, 0, 0]] * runner._compiled.layout.ranks,
            dtype=torch.int32,
        ),
    )
    main_ids = torch.tensor([[11, 12]], dtype=torch.long)
    main_pre_hc = torch.arange(
        runner._compiled.layout.ranks * runner._compiled.layout.decode_seq * 4,
        dtype=torch.float32,
    ).reshape(runner._compiled.layout.ranks, runner._compiled.layout.decode_seq, 4, 1)
    tail_hidden = torch.full((4, 1), 99.0, dtype=torch.float32)
    runner._mtp_request_states["req-a"] = SimpleNamespace(
        tail_token_id=9,
        tail_pre_hc_hidden=tail_hidden,
        tail_position=6,
    )

    committed_ids, committed_hidden, committed_positions = runner._mtp_committed_window(
        inputs,
        main_ids,
        main_pre_hc,
        request_index=0,
        accepted_count=1,
    )

    assert committed_ids.tolist() == [9, 11]
    assert committed_positions.tolist() == [6, 7]
    torch.testing.assert_close(committed_hidden[0], tail_hidden)
    torch.testing.assert_close(committed_hidden[1], main_pre_hc[0, 0])

    committed_ids, committed_hidden, committed_positions = runner._mtp_committed_window(
        inputs,
        main_ids,
        main_pre_hc,
        request_index=0,
        accepted_count=2,
    )

    assert committed_ids.tolist() == [11, 12]
    assert committed_positions.tolist() == [7, 8]
    torch.testing.assert_close(committed_hidden, main_pre_hc[0])


def test_cli_selects_deepseek_executor_and_forces_prefix_cache_off(tmp_path):
    model_dir = _write_deepseek_model_dir(tmp_path)
    args = cli.build_parser().parse_args(
        [
            "--model", str(model_dir),
            "--devices", "0,1,2,3,4,5,6,7",
            "--dp", "8",
            "--ep", "8",
            "--tp", "1",
            "--block-size", "128",
            "--max-model-len", "260",
            "--dtype", "int8",
            "--enable-mtp",
        ]
    )

    config = cli.build_serving_engine_config(args)

    assert config.executor_cls == "PyptoDeepSeekV4Executor"
    assert config.device_ids == (0, 1, 2, 3, 4, 5, 6, 7)
    assert config.parallel_config.replica_device_groups == ((0, 1, 2, 3, 4, 5, 6, 7),)
    assert config.runtime_config.page_size == 128
    assert config.runtime_config.weight_dtype == "int8"
    assert config.enable_prefix_cache is False
    assert config.executor_kwargs["enable_mtp"] is True


def test_tokenizer_falls_back_when_deepseek_config_fails_strict_validation(tmp_path, monkeypatch):
    class StrictDataclassFieldValidationError(Exception):
        pass

    class AutoTokenizer:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            raise StrictDataclassFieldValidationError("attention_dropout expected float")

    sentinel = object()
    fake_transformers = type(
        "FakeTransformers",
        (),
        {"AutoTokenizer": AutoTokenizer, "PreTrainedTokenizerFast": object},
    )
    fake_hub_errors = ModuleType("huggingface_hub.errors")
    fake_hub_errors.StrictDataclassFieldValidationError = StrictDataclassFieldValidationError
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "huggingface_hub.errors", fake_hub_errors)
    monkeypatch.setattr(tokenizer_module, "_load_fast_tokenizer_from_file", lambda *args: sentinel)
    (tmp_path / "tokenizer.json").touch()

    adapter = tokenizer_module.TransformersTokenizerAdapter.from_pretrained(str(tmp_path))

    assert adapter.tokenizer is sentinel


def test_deepseek_compile_attaches_lazy_weight_store_without_opening_shards(tmp_path, monkeypatch):
    model_dir = _write_deepseek_model_dir(tmp_path)
    kernel_dir = _write_deepseek_kernel_dir(tmp_path, lm_head_tp_size=8)
    monkeypatch.setattr(
        model_loader,
        "load_tokenizer",
        lambda *args, **kwargs: _Tokenizer(),
    )
    opened: list[Path] = []

    def _fail_open(path: Path, device: str):
        opened.append(path)
        raise AssertionError(f"unexpected safetensors open on {device}: {path}")

    monkeypatch.setattr(weight_loader, "_default_safe_open", _fail_open)
    monkeypatch.setattr(npu_executor, "_find_pypto_lib_deepseek_v4_dir", lambda *args, **kwargs: kernel_dir)
    loaded = ModelLoader().load(
        model_id="dsv4",
        model_dir=str(model_dir),
        runtime_config=RuntimeConfig(page_size=128, max_batch_size=4, max_seq_len=256, weight_dtype="int8"),
    )
    executor = npu_executor.DeepSeekV4PyptoExecutor(platform="a2a3sim", device_ids=tuple(range(8)))

    compiled = executor._compile_model(loaded.runtime_model)

    assert opened == []
    assert isinstance(compiled.weight_store, DeepSeekV4WeightStore)
    assert compiled.weight_store.filename_for("head.weight") == "model-00001-of-00001.safetensors"
    assert compiled.weight_store.device == "cpu"
    assert compiled.layer_plan[0].attention_kind == "swa"
    assert compiled.layer_plan[2].attention_kind == "csa"
    assert compiled.layer_plan[2].include_tid2eid is True
    assert compiled.layer_plan[3].attention_kind == "hca"
    assert compiled.layer_plan[3].include_gate_bias is True


def test_deepseek_compile_uses_signature_metadata_and_mtp_scalars(tmp_path, monkeypatch):
    model_dir = _write_deepseek_model_dir(tmp_path)
    kernel_dir = _write_deepseek_kernel_dir(tmp_path, lm_head_tp_size=8)
    monkeypatch.setattr(model_loader, "load_tokenizer", lambda *args, **kwargs: _Tokenizer())
    monkeypatch.setattr(npu_executor, "_find_pypto_lib_deepseek_v4_dir", lambda *args, **kwargs: kernel_dir)
    monkeypatch.setattr(DeepSeekV4WeightStore, "validate_mtp_startup_contract", lambda *args, **kwargs: None)
    loaded = ModelLoader().load(
        model_id="dsv4",
        model_dir=str(model_dir),
        runtime_config=RuntimeConfig(page_size=128, max_batch_size=4, max_seq_len=256, weight_dtype="int8"),
    )

    prefill_fwd = SimpleNamespace(l3_prefill_fwd=object())
    decode_fwd = SimpleNamespace(l3_decode_fwd=object())
    prefill_mtp = SimpleNamespace(l3_mtp_prefill_fwd=object())
    decode_mtp = SimpleNamespace(l3_mtp_decode_layer=object())
    compile_calls: list[tuple[str, object, dict[str, int] | None]] = []

    def _fake_compile(self, name, jit_fn, *, scalar_args=None):
        compile_calls.append((name, jit_fn, scalar_args))
        return DeepSeekV4L3Callable(compiled=object(), name=name)

    monkeypatch.setattr(
        npu_executor.DeepSeekV4PyptoExecutor,
        "_load_kernel_modules",
        lambda self, layout: {
            "config": object(),
            "prefill_fwd": prefill_fwd,
            "decode_fwd": decode_fwd,
            "prefill_mtp": prefill_mtp,
            "decode_mtp": decode_mtp,
            "rope_tables": object(),
        },
    )
    monkeypatch.setattr(npu_executor.DeepSeekV4PyptoExecutor, "_compile_l3_callable", _fake_compile)
    monkeypatch.setattr(
        npu_executor.DeepSeekV4PyptoExecutor,
        "_build_rope_tables",
        lambda self, rope_tables_module, config_module: (torch.empty(1), torch.empty(1)),
    )
    executor = npu_executor.DeepSeekV4PyptoExecutor(
        platform="a2a3sim",
        device_ids=tuple(range(8)),
        compile_kernels=True,
        enable_mtp=True,
    )

    compiled = executor._compile_model(loaded.runtime_model)

    assert compile_calls == [
        ("deepseek_v4_prefill", prefill_fwd.l3_prefill_fwd, None),
        ("deepseek_v4_decode", decode_fwd.l3_decode_fwd, None),
        ("deepseek_v4_mtp_prefill", prefill_mtp.l3_mtp_prefill_fwd, {"num_tokens": 128}),
        ("deepseek_v4_mtp_decode", decode_mtp.l3_mtp_decode_layer, {"num_tokens": 8}),
    ]
    assert compiled.prefill is not None
    assert compiled.decode is not None
    assert compiled.mtp_prefill is not None
    assert compiled.mtp_decode is not None


def test_deepseek_l3_compile_forwards_only_signature_scalars(monkeypatch):
    import pypto.ir.distributed_compiled_program as distributed_program_module
    import pypto.runtime as runtime_module

    class _DistributedCompiledProgram:
        pass

    class _DistributedConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class _RunConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr(
        distributed_program_module,
        "DistributedCompiledProgram",
        _DistributedCompiledProgram,
    )
    monkeypatch.setattr(distributed_program_module, "DistributedConfig", _DistributedConfig)
    monkeypatch.setattr(runtime_module, "RunConfig", _RunConfig)

    executor = npu_executor.DeepSeekV4PyptoExecutor.__new__(npu_executor.DeepSeekV4PyptoExecutor)
    executor._device_ids = tuple(range(8))
    executor._run_config = lambda *, codegen_only: SimpleNamespace(
        platform="a2a3",
        device_id=0,
        backend_type="pto",
        strategy=None,
        dump_passes=False,
        save_kernels=False,
        save_kernels_dir=None,
        diagnostic_phase=None,
        disabled_diagnostics=(),
        compile_profiling=False,
    )
    compile_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    class _JitFunction:
        @staticmethod
        def compile(*args, **kwargs):
            compile_calls.append((args, kwargs))
            return _DistributedCompiledProgram()

    compiled = executor._compile_l3_callable(
        "deepseek_v4_mtp_prefill",
        _JitFunction(),
        scalar_args={"num_tokens": 128},
    )

    assert isinstance(compiled.compiled, _DistributedCompiledProgram)
    assert len(compile_calls) == 1
    args, kwargs = compile_calls[0]
    assert args == ()
    assert kwargs["num_tokens"] == 128
    assert set(kwargs) == {"config", "num_tokens"}
    assert isinstance(kwargs["config"], _RunConfig)


def test_deepseek_kernel_contract_rejects_config_dimension_mismatch(tmp_path):
    kernel_dir = _write_deepseek_kernel_dir(tmp_path, lm_head_tp_size=8, block_size=64)
    executor = npu_executor.DeepSeekV4PyptoExecutor.__new__(npu_executor.DeepSeekV4PyptoExecutor)
    executor._kernel_dir = kernel_dir

    with pytest.raises(ValueError, match="BLOCK_SIZE=64 expected 128"):
        executor._validate_kernel_contract(DeepSeekV4CacheLayout())


def test_deepseek_kernel_contract_rejects_prefill_state_mismatch(tmp_path):
    kernel_dir = _write_deepseek_kernel_dir(
        tmp_path,
        lm_head_tp_size=8,
        hca_state_blocks=1024,
        csa_state_blocks=2048,
        csa_inner_state_blocks=2048,
    )
    executor = npu_executor.DeepSeekV4PyptoExecutor.__new__(npu_executor.DeepSeekV4PyptoExecutor)
    executor._kernel_dir = kernel_dir

    with pytest.raises(
        ValueError,
        match=(
            r"prefill_attention_hca.py:HCA_STATE_MAX_BLOCKS=1024 expected 2048"
            r".*prefill_attention_csa.py:CSA_STATE_MAX_BLOCKS=2048 expected 4096"
            r".*prefill_attention_csa.py:INNER_STATE_MAX_BLOCKS=2048 expected 4096"
        ),
    ):
        executor._validate_kernel_contract(DeepSeekV4CacheLayout())


def test_deepseek_weight_store_reads_real_safetensors_by_name(tmp_path):
    from safetensors.torch import save_file

    save_file(
        {
            "embed.weight": torch.arange(4, dtype=torch.float32).reshape(2, 2),
            "head.weight": torch.ones(2, 2),
        },
        str(tmp_path / "global.safetensors"),
    )
    store = DeepSeekV4WeightStore(
        model_dir=tmp_path,
        weight_map={
            "embed.weight": "global.safetensors",
            "head.weight": "global.safetensors",
        },
    )

    loaded = store.load_tensor("embed.weight")

    assert loaded.tolist() == [[0.0, 1.0], [2.0, 3.0]]


def test_deepseek_executor_lazily_loads_and_caches_embeddings(tmp_path):
    from safetensors.torch import save_file

    save_file(
        {"embed.weight": torch.arange(24, dtype=torch.float32).reshape(6, 4)},
        str(tmp_path / "embed.safetensors"),
    )
    open_count = 0
    store = DeepSeekV4WeightStore(
        model_dir=tmp_path,
        weight_map={"embed.weight": "embed.safetensors"},
    )
    original_open = store._safe_open_fn

    def _counting_open(path: Path, device: str):
        nonlocal open_count
        open_count += 1
        return original_open(path, device)

    store._safe_open_fn = _counting_open
    executor = npu_executor.DeepSeekV4PyptoExecutor.__new__(npu_executor.DeepSeekV4PyptoExecutor)
    executor._compiled = {
        "dsv4": DeepSeekV4CompiledKernels(
            layout=DeepSeekV4CacheLayout(),
            model_dir=str(tmp_path),
            weight_map=store.weight_map,
            weight_store=store,
            compress_ratios=tuple([0] * 44),
            layer_plan=build_deepseek_v4_layer_plan(
                compress_ratios=tuple([0] * 44),
                num_hidden_layers=43,
                num_hash_layers=3,
            ),
            kernel_dir=str(tmp_path),
        )
    }
    executor._embedding_cache = {}
    model = _runtime_model_for_embeddings()

    first = executor.lookup_embeddings(model, torch.tensor([1, 3], dtype=torch.long))
    second = executor.lookup_embeddings(model, torch.tensor([[2, 4]], dtype=torch.long))
    runner = DeepSeekV4ModelRunner(compiled=executor._compiled["dsv4"])
    runner_rows = runner._embedding_rows(torch.tensor([0, 5]), torch.float32)

    assert first.tolist() == [[4.0, 5.0, 6.0, 7.0], [12.0, 13.0, 14.0, 15.0]]
    assert second.shape == (1, 2, 4)
    assert second[0, 1].tolist() == [16.0, 17.0, 18.0, 19.0]
    assert runner_rows.tolist() == [[0.0, 1.0, 2.0, 3.0], [20.0, 21.0, 22.0, 23.0]]
    assert open_count == 1


def test_deepseek_weight_store_loads_rank_local_experts(tmp_path):
    core_names = deepseek_v4_layer_core_weight_names(0, include_tid2eid=True)
    local_experts = deepseek_v4_local_expert_ids(rank=1, ranks=4, n_routed_experts=8)
    expert_names = deepseek_v4_routed_expert_weight_names(0, local_experts)
    weight_map = {name: "layer.safetensors" for name in (*core_names, *expert_names)}
    (tmp_path / "layer.safetensors").touch()
    reads: list[str] = []

    class _Reader:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def get_tensor(self, name: str) -> torch.Tensor:
            reads.append(name)
            return torch.tensor([len(reads)])

    store = DeepSeekV4WeightStore(model_dir=tmp_path, weight_map=weight_map, safe_open_fn=lambda path, device: _Reader())

    loaded = store.load_rank_layer_weights(
        0,
        rank=1,
        ranks=4,
        n_routed_experts=8,
        include_tid2eid=True,
    )

    assert local_experts == (2, 3)
    assert set(loaded) == set(weight_map)
    assert all(".experts.2." in name or ".experts.3." in name for name in expert_names)
    assert not any(".experts.0." in name or ".experts.1." in name for name in loaded)


def test_deepseek_weight_store_packs_lm_head_into_8_tp_shards(tmp_path):
    from safetensors.torch import save_file

    save_file(
        {
            "embed.weight": torch.arange(64, dtype=torch.float32).reshape(16, 4),
            "norm.weight": torch.arange(4, dtype=torch.float32),
            "head.weight": torch.arange(64, dtype=torch.float32).reshape(16, 4) + 100,
            "hc_head_fn": torch.zeros((4, 16), dtype=torch.float32),
            "hc_head_scale": torch.ones((1,), dtype=torch.float32),
            "hc_head_base": torch.zeros((4,), dtype=torch.float32),
        },
        str(tmp_path / "global.safetensors"),
    )
    store = DeepSeekV4WeightStore(
        model_dir=tmp_path,
        weight_map={
            "embed.weight": "global.safetensors",
            "norm.weight": "global.safetensors",
            "head.weight": "global.safetensors",
            "hc_head_fn": "global.safetensors",
            "hc_head_scale": "global.safetensors",
            "hc_head_base": "global.safetensors",
        },
    )

    global_weights = store.load_packed_global_weights(ranks=8)

    assert global_weights.lm_head_layout.vocab_per_rank == 2
    assert global_weights.lm_head_layout.padded_vocab_per_rank == 512
    assert global_weights.lm_head_weight.shape == (8, 512, 4)
    assert global_weights.lm_head_weight[0, :2].tolist() == [[100.0, 101.0, 102.0, 103.0], [104.0, 105.0, 106.0, 107.0]]
    assert global_weights.lm_head_weight[1, :2].tolist() == [[108.0, 109.0, 110.0, 111.0], [112.0, 113.0, 114.0, 115.0]]
    assert torch.count_nonzero(global_weights.lm_head_weight[:, 2:]) == 0


def test_deepseek_layer_packer_transposes_and_stacks_rank_local_experts():
    raw = _synthetic_layer_raw(layer_id=0, n_experts=4)

    packed = pack_deepseek_v4_layer_weights(
        0,
        raw,
        ranks=2,
        n_routed_experts=4,
        compress_ratio=4,
        include_tid2eid=False,
        include_gate_bias=True,
    )

    assert packed.tensors["wq_a"].shape == (2, 4, 2)
    assert packed.tensors["wq_a"][0].tolist() == raw["layers.0.attn.wq_a.weight"].t().tolist()
    assert packed.tensors["wo_a"].shape == (2, 8, 2, 4)
    assert packed.tensors["csa_cmp_wkv"].shape == (2, 2, 4)
    assert packed.tensors["csa_cmp_wkv"][0].tolist() == raw["layers.0.attn.compressor.wkv.weight"].tolist()
    assert packed.tensors["csa_inner_wkv"].shape == (2, 2, 4)
    assert packed.tensors["csa_inner_wkv"][0].tolist() == raw["layers.0.attn.indexer.compressor.wkv.weight"].tolist()
    assert packed.tensors["hca_cmp_wkv"].shape == (2, 512, 4096)
    assert torch.count_nonzero(packed.tensors["hca_cmp_wkv"]) == 0
    assert packed.tensors["gate_bias"].shape == (2, 4)
    assert packed.tensors["tid2eid"].shape == (2, 129280, 6)
    assert packed.tensors["routed_w1"].shape == (2, 2, 2, 4)
    assert packed.tensors["routed_w1"][0, 0].tolist() == raw["layers.0.ffn.experts.0.w1.weight"].tolist()
    assert packed.tensors["routed_w1"][1, 0].tolist() == raw["layers.0.ffn.experts.2.w1.weight"].tolist()
    assert torch.equal(packed.tensors["csa_hadamard_idx"][0], deepseek_v4_hadamard_idx())

    destination_storage = {
        name: torch.empty(
            (int(tensor.shape[0]), int(tensor.shape[1]) * 2, *tensor.shape[2:]),
            dtype=tensor.dtype,
        )
        for name, tensor in packed.tensors.items()
    }
    destinations = {
        name: storage[:, int(packed.tensors[name].shape[1]) :]
        for name, storage in destination_storage.items()
    }
    assert not all(destination.is_contiguous() for destination in destinations.values())
    direct = pack_deepseek_v4_layer_weights(
        0,
        raw,
        ranks=2,
        n_routed_experts=4,
        compress_ratio=4,
        include_tid2eid=False,
        include_gate_bias=True,
        destinations=destinations,
    )

    assert direct.tensors.keys() == packed.tensors.keys()
    for name, expected in packed.tensors.items():
        assert direct.tensors[name] is destinations[name]
        assert torch.equal(direct.tensors[name], expected), name


def test_deepseek_stacked_weight_loader_packs_subsequent_layers_into_final_slices(monkeypatch):
    def packed_layer(layer_id: int) -> weight_loader.DeepSeekV4PackedLayerWeights:
        tensors = {"fwd": torch.full((2, 2), layer_id, dtype=torch.int8)}
        tensors.update(
            {
                name: torch.full((2, 1), layer_id * 20 + index, dtype=torch.float32)
                for index, name in enumerate(weight_loader.DEEPSEEK_V4_CSA_STACKED_WEIGHT_NAMES)
            }
        )
        tensors.update(
            {
                name: torch.full((2, 1), layer_id * 20 + 12 + index, dtype=torch.float32)
                for index, name in enumerate(weight_loader.DEEPSEEK_V4_HCA_STACKED_WEIGHT_NAMES)
            }
        )
        return weight_loader.DeepSeekV4PackedLayerWeights(layer_id=layer_id, tensors=tensors)

    layers = [packed_layer(layer_id) for layer_id in range(3)]
    direct_flags: list[bool] = []
    store = DeepSeekV4WeightStore(model_dir=".", weight_map={})

    def fake_load(layer_id: int, **kwargs):
        destinations = kwargs.get("destinations")
        direct_flags.append(destinations is not None)
        packed = layers[layer_id]
        if destinations is None:
            return packed
        for name, destination in destinations.items():
            destination.copy_(packed.tensors[name])
        return weight_loader.DeepSeekV4PackedLayerWeights(layer_id=layer_id, tensors=destinations)

    monkeypatch.setattr(store, "load_packed_layer_weights", fake_load)
    monkeypatch.setattr(torch, "cat", lambda *args, **kwargs: pytest.fail("torch.cat must not be used"))

    stacked = store.load_stacked_layer_weights(
        ranks=2,
        n_routed_experts=4,
        compress_ratios=(0, 4, 128),
        num_hash_layers=1,
    )

    assert direct_flags == [False, True, True]
    assert stacked.tensors["fwd"].tolist() == [[0, 0, 1, 1, 2, 2], [0, 0, 1, 1, 2, 2]]
    for name in weight_loader.DEEPSEEK_V4_CSA_STACKED_WEIGHT_NAMES:
        assert torch.equal(stacked.tensors[name], layers[1].tensors[name])
    for name in weight_loader.DEEPSEEK_V4_HCA_STACKED_WEIGHT_NAMES:
        assert torch.equal(stacked.tensors[name], layers[2].tensors[name])
    assert all(tensor.is_contiguous() for tensor in stacked.tensors.values())


def test_deepseek_retain_stacked_host_weights_keeps_ordinary_storage():
    runner = DeepSeekV4ModelRunner.__new__(DeepSeekV4ModelRunner)
    runner._stacked_host_weights = None
    runner._l3_worker = None
    weight = torch.arange(8, dtype=torch.float32).reshape(2, 4)

    retained = runner._retain_stacked_host_weights(
        weight_loader.DeepSeekV4StackedLayerWeights(tensors={"weight": weight})
    )

    assert retained.tensors["weight"] is weight
    assert not retained.tensors["weight"].is_shared()


def test_deepseek_worker_registers_main_and_mtp_weights_for_inheritance(monkeypatch):
    main_weight = torch.zeros((1, 2), dtype=torch.float32)
    mtp_weight = torch.ones((1, 2), dtype=torch.float32)
    compiled_program = object()
    captured = {}

    class FakeDistributedWorker:
        def __init__(self, compiled, *, persistent, inherited_host_tensors):
            captured["compiled"] = compiled
            captured["persistent"] = persistent
            captured["inherited"] = inherited_host_tensors

    monkeypatch.setattr("pypto.runtime.DistributedWorker", FakeDistributedWorker)
    runner = DeepSeekV4ModelRunner.__new__(DeepSeekV4ModelRunner)
    runner._l3_worker = None
    runner._stacked_host_weights = {"main": main_weight}
    runner._mtp_buffers = type("MtpBuffers", (), {"weights": {"mtp": mtp_weight}})()
    runner._compiled = type(
        "Compiled",
        (),
        {"l3_callables": lambda _self: (DeepSeekV4L3Callable(compiled_program, "decode"),)},
    )()
    runner._assert_l3_shared_buffers_preallocated = lambda: None

    worker = runner._shared_l3_worker()

    assert isinstance(worker, FakeDistributedWorker)
    assert captured["compiled"] == [compiled_program]
    assert captured["persistent"] is True
    assert captured["inherited"] == [main_weight, mtp_weight]


def test_deepseek_resident_upload_releases_inherited_host_references():
    main_weight = torch.zeros((1, 2), dtype=torch.float32)
    mtp_weight = torch.ones((1, 2), dtype=torch.float32)

    class FakeWorker:
        def __init__(self):
            self.released = False

        def alloc_stacked_tensor(self, tensor):
            return tensor

        def free_stacked_tensor(self, _tensor):
            pass

        def release_inherited_host_tensor_refs(self):
            self.released = True

    worker = FakeWorker()
    runner = DeepSeekV4ModelRunner.__new__(DeepSeekV4ModelRunner)
    runner._stacked_host_weights = {"main": main_weight}
    runner._stacked_device_weights = None
    runner._mtp_buffers = type("MtpBuffers", (), {"weights": {"mtp": mtp_weight}})()
    runner._mtp_device_weights = None
    runner._shared_l3_worker = lambda: worker

    runner._materialize_resident_weights()

    assert worker.released
    assert runner._stacked_host_weights is None
    assert not runner._mtp_buffers.weights


def test_deepseek_cache_metadata_maps_scheduler_block_ids():
    metadata = DeepSeekV4CacheMetadataBuilder(layout=DeepSeekV4CacheLayout())

    table = metadata.block_table_from_ids([[64, 65]], max_blocks=4)
    assert table.tolist() == [[64, 65, 0, 0]]

    cmp_mapping = metadata.slot_mapping_from_ids(
        [[64]],
        [[0, 4, 256]],
        block_size=128,
        compress_ratio=4,
    )
    base = 64 * 128
    assert cmp_mapping.tolist() == [[base, base + 1, base + 64]]

    hca_state_mapping = metadata.slot_mapping_from_ids(
        [[64]],
        [[0, 128, 256]],
        block_size=8,
        compress_ratio=128,
    )
    assert hca_state_mapping.tolist() == [[64 * 8, 64 * 8 + 1, 64 * 8 + 2]]


def test_deepseek_cache_group_specs_leave_physical_capacity_for_runtime_sizing():
    compress_ratios = (0, 0, *([4] * 21), *([128] * 20))
    specs = build_deepseek_v4_cache_group_specs(43, compress_ratios, decode_batch=8)
    by_name = {spec.name: spec for spec in specs}

    assert all(spec.num_blocks is None for spec in specs)
    assert by_name["ori"].spec.page_size_bytes == 43 * 128 * 512 * 2
    assert by_name["idx"].spec.page_size_bytes == 21 * 128 * (128 + 4)
    assert by_name["hca_state"].spec.page_size_bytes == 20 * 8 * 1024 * 4
    assert deepseek_v4_cache_blocks_for_slots(specs, 3) == {
        "ori": 96,
        "cmp": 24,
        "idx": 48,
        "hca_state": 48,
        "csa_state": 48,
        "csa_inner_state": 48,
    }


def test_deepseek_cache_sizing_uses_limiting_rank_post_weight_budget(monkeypatch):
    layout = DeepSeekV4CacheLayout(decode_batch=8, decode_seq=1, decode_tokens=8)
    runner = DeepSeekV4ModelRunner(
        compiled=DeepSeekV4CompiledKernels(
            layout=layout,
            model_dir="",
            weight_map={},
            weight_store=None,
            compress_ratios=tuple([0] * 43),
            layer_plan=(),
            kernel_dir="",
            device_id=2,
            device_ids=(2, 5),
        )
    )
    runner._cache_group_specs = build_deepseek_v4_cache_group_specs(
        43,
        runner._compiled.compress_ratios,
        decode_batch=layout.decode_batch,
    )
    memory = {
        "npu:2": (5_000_000_000, 10_000_000_000),
        "npu:5": (4_000_000_000, 10_000_000_000),
    }
    monkeypatch.setattr(torch.npu, "mem_get_info", lambda device: memory[device])
    runtime = RuntimeConfig(npu_memory_utilization=0.8)

    bytes_per_slot = sum(
        spec.max_blocks_per_seq * spec.spec.page_size_bytes
        for spec in runner._cache_group_specs
    )
    scratch_bytes = sum(
        layout.decode_batch * spec.spec.page_size_bytes
        for spec in runner._cache_group_specs
    )
    expected = max((2_000_000_000 - scratch_bytes) // bytes_per_slot, 1)

    assert runner._compute_kv_cache_capacity_slots(runtime) == expected


def test_deepseek_cache_allocation_halves_all_groups_together_on_oom(monkeypatch):
    layout = DeepSeekV4CacheLayout(decode_batch=8, decode_seq=1, decode_tokens=8)
    runner = DeepSeekV4ModelRunner(
        compiled=DeepSeekV4CompiledKernels(
            layout=layout,
            model_dir="",
            weight_map={},
            weight_store=None,
            compress_ratios=tuple([0] * 43),
            layer_plan=(),
            kernel_dir="",
        )
    )
    runner._cache_group_specs = build_deepseek_v4_cache_group_specs(
        43,
        runner._compiled.compress_ratios,
        decode_batch=layout.decode_batch,
    )
    attempts = []

    def allocate_main_cache():
        slots = runner._cache_group_num_blocks["ori"] // 32
        attempts.append(slots)
        if slots > 2:
            raise MemoryError("synthetic OOM")
        return object()

    monkeypatch.setattr(runner, "_materialize_decode_device_cache", allocate_main_cache)
    monkeypatch.setattr(runner, "_materialize_mtp_device_kv_cache", lambda: None)

    assert runner._alloc_kv_cache_with_retry(8) == 2
    assert attempts == [8, 4, 2]
    assert runner._cache_group_num_blocks == deepseek_v4_cache_blocks_for_slots(
        runner._cache_group_specs,
        2,
    )


def test_deepseek_device_cache_allocates_runtime_sized_rank_shards():
    layout = DeepSeekV4CacheLayout(
        ranks=2,
        block_size=1,
        decode_batch=1,
        decode_seq=1,
        decode_tokens=1,
    )
    runner = DeepSeekV4ModelRunner(
        compiled=DeepSeekV4CompiledKernels(
            layout=layout,
            model_dir="",
            weight_map={},
            weight_store=None,
            compress_ratios=tuple([0] * 43),
            layer_plan=(),
            kernel_dir="",
        )
    )
    runner._cache_group_num_blocks = {
        name: 2
        for name in ("ori", "cmp", "idx", "hca_state", "csa_state", "csa_inner_state")
    }

    class FakeWorker:
        def __init__(self):
            self.allocations = []
            self.frees = []

        def alloc_tensor(self, shape, dtype, *, worker_id=0):
            tensor = DeviceTensor(
                0x1000 + len(self.allocations) * 0x100,
                tuple(shape),
                dtype,
            )
            self.allocations.append((worker_id, tensor))
            return tensor

        def free_tensor(self, tensor, *, worker_id=0):
            self.frees.append((worker_id, tensor))

        def free_stacked_tensor(self, stacked):
            for tensor, worker_id in zip(
                stacked.shards,
                stacked.worker_ids,
                strict=True,
            ):
                self.free_tensor(tensor, worker_id=worker_id)

    worker = FakeWorker()
    runner._l3_worker = worker

    cache = runner._materialize_decode_device_cache()

    assert cache.kv_cache.full_shape == (2, 43 * 3, 1, 1, 512)
    assert cache.cmp_kv.full_shape == (2, 43 * 3, 1, 1, 512)
    assert cache.idx_kv_cache.full_shape == (2, 21 * 3, 1, 1, 128)
    assert cache.hca_compress_state.full_shape == (2, 20 * 3, 8, 1024)
    assert len(worker.allocations) == 14
    assert {worker_id for worker_id, _tensor in worker.allocations} == {0, 1}

    runner._free_device_caches()
    assert len(worker.frees) == 14


def _grouped_cache_rows(count: int) -> list[dict[str, list[int]]]:
    names = ("ori", "cmp", "idx", "hca_state", "csa_state", "csa_inner_state")
    return [
        {name: [request_index * len(names) + group_index] for group_index, name in enumerate(names)}
        for request_index in range(count)
    ]


def test_deepseek_prepare_prefill_inputs_maps_chunk_metadata():
    runner, model = _runner_for_prepared_inputs()
    layout = runner._compiled.layout
    embeddings = torch.arange(12, dtype=torch.bfloat16).reshape(3, 4)

    prepared = runner.prepare_prefill_inputs(
        model,
        PrefillBatch(
            request_ids=["req-a"],
            token_ids=torch.tensor([10, 11, 12], dtype=torch.long),
            input_embeddings=embeddings,
            seq_lens=[129],
            chunk_lens=[3],
            chunk_offsets=[0],
            chunk_starts=[126],
            block_ids_by_group=_grouped_cache_rows(1),
            cache_partitions=[0],
        ),
    )

    assert prepared.request_ids == ("req-a",)
    assert prepared.ranks == (0,)
    assert prepared.actual_tokens == (3,)
    assert prepared.x_hc.shape == (8, 128, 4, 4)
    assert prepared.x_hc.dtype == torch.float32
    assert prepared.ori_block_table.shape == (8, 128)
    assert prepared.ori_block_table[0, :4].tolist() == [0, 0, 0, 0]
    assert prepared.cmp_block_table.shape == (8, 32)
    assert prepared.idx_block_table.shape == (8, 64)
    assert prepared.position_ids.shape == (8, 128)
    assert prepared.position_ids[0, :4].tolist() == [126, 127, 128, 129]
    assert prepared.input_ids[0, :4].tolist() == [10, 11, 12, 10]
    assert prepared.ori_slot_mapping.shape == (8, 128)
    assert prepared.ori_slot_mapping[0, :4].tolist() == [126, 127, 0, -1]
    assert prepared.hca_cmp_slot_mapping.shape == (8, 128)
    assert prepared.hca_cmp_slot_mapping[0, :3].tolist() == [-1, 128, -1]
    assert prepared.hca_cmp_slot_mapping[0, 3].item() == -1
    assert prepared.csa_cmp_slot_mapping.shape == (8, 128)
    assert prepared.csa_cmp_slot_mapping[0, :3].tolist() == [-1, 159, -1]
    assert prepared.csa_cmp_slot_mapping[0, 3].item() == -1
    assert prepared.csa_idx_slot_mapping.shape == (8, 128)
    assert prepared.csa_idx_slot_mapping[0, :3].tolist() == [-1, 287, -1]
    assert prepared.csa_idx_slot_mapping[0, 3].item() == -1
    assert prepared.hca_state_slot_mapping.shape == (8, 128)
    assert prepared.hca_state_slot_mapping[0, :4].tolist() == [
        30,
        31,
        24,
        -1,
    ]
    assert prepared.csa_state_slot_mapping.shape == (8, 128)
    assert prepared.csa_state_slot_mapping[0, :4].tolist() == [
        18,
        19,
        16,
        -1,
    ]
    assert prepared.csa_inner_state_slot_mapping.shape == (8, 128)
    assert prepared.csa_inner_state_slot_mapping[0, :4].tolist() == [
        22,
        23,
        20,
        -1,
    ]
    assert prepared.num_tokens_per_owner.tolist() == [3, 0, 0, 0, 0, 0, 0, 0]
    assert prepared.logit_row_indices[0].tolist() == [2, -1, -1, -1, -1, -1, -1, -1]


def test_deepseek_prepare_decode_inputs_requires_hidden_states():
    runner, model = _runner_for_prepared_inputs()

    with pytest.raises(ValueError, match="DeepSeek V4 decode requires host hidden states"):
        runner.prepare_decode_inputs(
            model,
            DecodeBatch(
                request_ids=["req-a"],
                token_ids=torch.tensor([[5]], dtype=torch.long),
                hidden_states=None,
                seq_lens=torch.tensor([128], dtype=torch.int32),
            ),
        )


def test_deepseek_prepare_mtp_decode_inputs_builds_sliding_window_metadata():
    runner, model = _runner_for_prepared_inputs()

    prepared = runner.prepare_mtp_decode_inputs(
        model,
        DecodeBatch(
            request_ids=["req-a", "req-b"],
            token_ids=torch.tensor([[5], [9]], dtype=torch.long),
            hidden_states=torch.arange(8, dtype=torch.bfloat16).reshape(2, 4),
            seq_lens=torch.tensor([128, 5], dtype=torch.int32),
            block_ids_by_group=_grouped_cache_rows(2),
            cache_partitions=[0, 1],
            prev_token_ids=torch.tensor([3, 7], dtype=torch.long),
            prev_hidden_states=torch.arange(8, 16, dtype=torch.bfloat16).reshape(2, 4),
        ),
    )

    # Each request owns one DP rank and its scheduler-provided physical pages.
    assert prepared.block_table.shape == (8, 4, 128)
    assert prepared.block_table[0, 0, :4].tolist() == [0, 0, 0, 0]
    assert prepared.block_table[1, 0, :4].tolist() == [6, 6, 6, 6]
    # SWA full window includes the current position and lowers through the
    # request's physical ori page.
    assert prepared.swa_lens[0, 0].item() == 127
    assert prepared.swa_lens[0, 1].item() == 128
    assert prepared.swa_lens[1, 1].item() == 5
    assert prepared.swa_indices[0, 1, :4].tolist() == [0, 1, 2, 3]
    assert prepared.swa_indices[1, 1, :5].tolist() == [768, 769, 770, 771, 772]
    # Paged write slot for the current token.
    assert prepared.swa_slot_mapping[0, 1].item() == 127
    assert prepared.swa_slot_mapping[1, 1].item() == 772
    # HCA/CSA follow pypto-lib's cache-first contract: their window contains
    # historical rows and the current decode position after KV writeback.
    assert prepared.window_swa_lens[0, 0].item() == 127
    assert prepared.window_swa_lens[0, 1].item() == 128
    assert prepared.window_swa_lens[1, 1].item() == 5
    assert prepared.window_swa_indices[1, 1, :5].tolist() == [768, 769, 770, 771, 772]
    assert prepared.num_tokens_per_owner.tolist() == [2, 2, 0, 0, 0, 0, 0, 0]
    assert prepared.logit_row_indices[0].tolist() == [0, 1, -1, -1, -1, -1, -1, -1]


def test_deepseek_prepare_mtp_decode_inputs_feeds_two_real_tokens():
    runner, model = _runner_for_prepared_inputs()

    prepared = runner.prepare_mtp_decode_inputs(
        model,
        DecodeBatch(
            request_ids=["req-a", "req-b"],
            token_ids=torch.tensor([[5], [9]], dtype=torch.long),
            hidden_states=torch.arange(8, dtype=torch.bfloat16).reshape(2, 4),
            seq_lens=torch.tensor([128, 5], dtype=torch.int32),
            block_ids_by_group=_grouped_cache_rows(2),
            cache_partitions=[0, 1],
            prev_token_ids=torch.tensor([3, 7], dtype=torch.long),
            prev_hidden_states=torch.arange(8, 16, dtype=torch.bfloat16).reshape(2, 4),
        ),
    )

    # Active rows get [prev_token, last_token]; positions are (seq_len-2, seq_len-1).
    assert prepared.input_ids[0, :6].tolist() == [3, 5, 3, 5, 3, 5]
    assert prepared.input_ids[1, :2].tolist() == [7, 9]
    assert prepared.position_ids[0, :6].tolist() == [126, 127, 126, 127, 126, 127]
    assert prepared.position_ids[1, :2].tolist() == [3, 4]
    assert prepared.kv_seq_lens[0, :4].tolist() == [128, 128, 128, 128]
    assert prepared.kv_seq_lens[1, :4].tolist() == [5, 5, 5, 5]
    # slot 0 carries the prev-token embedding, slot 1 the last-token embedding.
    assert prepared.x_hc[0, 0, 0].tolist() == [8, 9, 10, 11]
    assert prepared.x_hc[0, 1, 0].tolist() == [0, 1, 2, 3]
    assert prepared.x_hc[1, 0, 0].tolist() == [12, 13, 14, 15]
    assert prepared.x_hc[1, 1, 0].tolist() == [4, 5, 6, 7]
    # Padding row keeps replicating row 0's last embedding.
    assert prepared.x_hc[0, 4, 0].tolist() == [0, 1, 2, 3]


def test_deepseek_stage_decode_inputs_uses_shared_buffers():
    runner, model = _runner_for_prepared_inputs()
    prepared = runner.prepare_decode_inputs(
        model,
        DecodeBatch(
            request_ids=["req-a"],
            token_ids=torch.tensor([[5]], dtype=torch.long),
            hidden_states=torch.arange(4, dtype=torch.bfloat16).reshape(1, 4),
            seq_lens=torch.tensor([128], dtype=torch.int32),
            block_ids_by_group=_grouped_cache_rows(1),
            cache_partitions=[0],
        ),
    )

    staged = runner._stage_decode_inputs(prepared)

    assert staged.x_hc.is_shared()
    assert staged.x_hc.dtype == torch.float32
    assert runner._decode_buffers is not None
    assert runner._decode_buffers.x_hc_b.is_shared()
    for name in (
        "input_ids",
        "position_ids",
        "kv_seq_lens",
        "block_table",
        "ori_slot_mapping",
        "window_swa_indices",
        "window_swa_lens",
        "swa_slot_mapping",
        "swa_indices",
        "swa_lens",
        "cmp_block_table",
        "idx_block_table",
        "hca_compress_state_block_table",
        "csa_compress_state_block_table",
        "csa_inner_compress_state_block_table",
        "hca_cmp_slot_mapping",
        "hca_state_slot_mapping",
        "csa_cmp_slot_mapping",
        "csa_idx_slot_mapping",
        "csa_state_slot_mapping",
        "csa_inner_state_slot_mapping",
    ):
        assert getattr(staged, name).is_shared()


def test_deepseek_run_decode_dispatches_active_token_count():
    runner, model = _runner_for_prepared_inputs()
    runner._compiled.decode = DeepSeekV4L3Callable(compiled=object(), name="decode")
    captured: dict[str, object] = {}

    def fake_stage(inputs):
        captured["prepared"] = inputs
        return inputs

    def fake_decode_fwd_args(inputs, x_hc, pre_hc_hidden_out, hidden_out, logits):
        captured["x_hc_shape"] = tuple(x_hc.shape)
        captured["num_tokens_per_owner"] = inputs.num_tokens_per_owner
        return (x_hc, pre_hc_hidden_out, hidden_out, logits)

    def fake_run_l3(_callable, *args):
        args[-1].fill_(1)

    hidden_out = torch.empty(
        runner._compiled.layout.ranks,
        runner._compiled.layout.decode_tokens,
        model.config.hidden_size,
        dtype=torch.bfloat16,
    )
    runner._ensure_l3_shared_buffers = lambda _model: None
    runner._stage_decode_inputs = fake_stage
    runner._require_decode_buffers = lambda: SimpleNamespace(
        x_hc_a=captured["prepared"].x_hc,
        pre_hc_hidden_out=torch.empty(
            runner._compiled.layout.ranks,
            runner._compiled.layout.decode_tokens,
            runner._compiled.layout.hc_mult,
            model.config.hidden_size,
            dtype=torch.float32,
        ),
    )
    runner._require_decode_output_buffer = lambda _hidden_size: hidden_out
    runner._require_decode_logits_buffer = lambda _vocab_size: torch.empty(
        runner._compiled.layout.ranks,
        runner._compiled.layout.decode_tokens,
        model.config.vocab_size,
        dtype=torch.float32,
    )
    runner._decode_fwd_args = fake_decode_fwd_args
    runner._run_l3 = fake_run_l3

    result = runner.run_decode(
        model,
        DecodeBatch(
            request_ids=["req-a"],
            token_ids=torch.tensor([[5]], dtype=torch.long),
            hidden_states=torch.arange(4, dtype=torch.bfloat16).reshape(1, 4),
            seq_lens=torch.tensor([128], dtype=torch.int32),
            block_ids_by_group=_grouped_cache_rows(1),
            cache_partitions=[0],
        ),
    )

    assert captured["num_tokens_per_owner"].tolist() == [1, 0, 0, 0, 0, 0, 0, 0]
    assert captured["x_hc_shape"] == (
        runner._compiled.layout.ranks,
        runner._compiled.layout.decode_tokens,
        runner._compiled.layout.hc_mult,
        model.config.hidden_size,
    )
    assert result.logits.shape == (1, model.config.vocab_size)


def test_deepseek_prefill_staging_keeps_worker_resident_cache_tensors_out():
    layout = DeepSeekV4CacheLayout(
        ranks=1,
        prefill_seq=1,
        decode_batch=1,
        decode_seq=1,
        decode_tokens=1,
        block_size=1,
        prefill_ori_max_blocks=1,
        decode_ori_max_blocks=1,
        ori_table_max_blocks=1,
        cmp_max_blocks=1,
        idx_max_blocks=1,
        hca_state_max_blocks=1,
        csa_state_max_blocks=1,
        csa_inner_state_max_blocks=1,
        c128_state_block_size=1,
        c4_state_block_size=1,
        prefill_cmp_max_blocks=1,
        prefill_idx_max_blocks=1,
        prefill_hca_state_max_blocks=1,
        prefill_csa_state_max_blocks=1,
        prefill_csa_inner_state_max_blocks=1,
    )
    runner = DeepSeekV4ModelRunner(
        compiled=DeepSeekV4CompiledKernels(
            layout=layout,
            model_dir="",
            weight_map={},
            weight_store=None,
            compress_ratios=(),
            layer_plan=(),
            kernel_dir="",
            freqs_cos=torch.empty((1, 1), dtype=torch.bfloat16),
            freqs_sin=torch.empty((1, 1), dtype=torch.bfloat16),
        )
    )
    prefill = runner._ensure_prefill_fwd_buffers(hidden_size=1)

    assert not hasattr(runner, "_decode_work_cache")
    for name in (
        "kv_cache",
        "cmp_kv",
        "idx_kv_cache",
        "idx_kv_scale",
        "hca_compress_state",
        "csa_compress_state",
        "csa_inner_compress_state",
    ):
        assert name not in prefill.tensors


def test_deepseek_shared_buffer_setup_does_not_require_cache_zeroing_buffers(monkeypatch):
    runner, model = _runner_for_prepared_inputs()
    runner._stacked_device_weights = object()

    for method_name in (
        "load_packed_global_weights",
        "_static_freqs_cos_tensor",
        "_static_freqs_sin_tensor",
        "_ensure_decode_buffers",
        "_ensure_mtp_buffers",
        "_require_prefill_output_buffer",
        "_require_prefill_pre_hc_output_buffer",
        "_require_prefill_logits_buffer",
        "_static_final_norm_weight_tensor",
        "_static_lm_head_weight_tensor",
        "_require_decode_logits_buffer",
        "_hc_head_tensors",
        "_ensure_prefill_fwd_buffers",
        "_assert_l3_shared_buffers_preallocated",
        "_materialize_resident_weights",
    ):
        monkeypatch.setattr(runner, method_name, lambda *args, **kwargs: None)

    runner._ensure_l3_shared_buffers(model)

    assert not hasattr(runner, "_decode_work_cache")
    assert not hasattr(runner, "_cache_zero_page")


def test_deepseek_mtp_prefill_and_decode_reuse_same_kv_cache():
    layout = DeepSeekV4CacheLayout(
        ranks=1,
        prefill_seq=1,
        decode_batch=1,
        decode_seq=1,
        decode_tokens=1,
        block_size=1,
        prefill_ori_max_blocks=1,
        decode_ori_max_blocks=1,
        sliding_window=1,
    )
    runner = DeepSeekV4ModelRunner(
        compiled=DeepSeekV4CompiledKernels(
            layout=layout,
            model_dir="",
            weight_map={},
            weight_store=None,
            compress_ratios=(),
            layer_plan=(),
            kernel_dir="",
            mtp_prefill=DeepSeekV4L3Callable(compiled=object(), name="mtp_prefill"),
            mtp_decode=DeepSeekV4L3Callable(compiled=object(), name="mtp_decode"),
        )
    )
    weight = torch.arange(2, dtype=torch.float32)
    runner.load_mtp_weights = lambda: weight_loader.DeepSeekV4MtpWeights(tensors={"weight": weight})

    buffers = runner._ensure_mtp_buffers(hidden_size=1)

    assert buffers is not None
    assert buffers.weights["weight"] is weight
    assert not buffers.weights["weight"].is_shared()
    assert buffers.prefill_kv_cache is buffers.decode_kv_cache


def test_deepseek_release_finished_requests_discards_mtp_state():
    runner, _model = _runner_for_prepared_inputs()
    runner._mtp_request_states = {
        "req-a": SimpleNamespace(proposed_tokens=0),
        "req-b": SimpleNamespace(proposed_tokens=0),
    }

    runner.release_finished_requests(["req-a"])

    assert runner._mtp_request_states == {
        "req-b": SimpleNamespace(proposed_tokens=0),
    }


def _write_deepseek_model_dir(tmp_path: Path, *, quant_method: str = "compressed-tensors") -> Path:
    model_dir = tmp_path / "dsv4-flash-w8a8"
    model_dir.mkdir()
    compress_ratios = _deepseek_flash_compress_ratios()
    config = {
        "architectures": ["DeepseekV4ForCausalLM"],
        "model_type": "deepseek_v4",
        "vocab_size": 129280,
        "hidden_size": 4096,
        "moe_intermediate_size": 2048,
        "n_routed_experts": 256,
        "n_shared_experts": 1,
        "num_hidden_layers": 43,
        "num_attention_heads": 64,
        "num_key_value_heads": 1,
        "head_dim": 512,
        "max_position_embeddings": 1048576,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000,
        "bos_token_id": 0,
        "eos_token_id": 1,
        "torch_dtype": "bfloat16",
        "compress_ratios": compress_ratios,
        "quantization_config": {
            "quant_method": quant_method,
            "format": "int-quantized",
            "quantization_status": "compressed",
        },
    }
    (model_dir / "config.json").write_text(json.dumps(config))
    weight_names = deepseek_v4_startup_weight_names(
        43,
        n_routed_experts=256,
        compress_ratios=compress_ratios,
        num_hash_layers=3,
    )
    index = {"weight_map": {name: "model-00001-of-00001.safetensors" for name in weight_names}}
    (model_dir / "model.safetensors.index.json").write_text(json.dumps(index))
    return model_dir


def _deepseek_flash_compress_ratios() -> list[int]:
    return [0, 0, *(4 if layer_id % 2 == 0 else 128 for layer_id in range(2, 43)), 0]


def _write_deepseek_kernel_dir(
    tmp_path: Path,
    *,
    lm_head_tp_size: int,
    use_config_constant: bool = False,
    block_size: int = 128,
    hca_state_blocks: int = 2048,
    csa_state_blocks: int = 4096,
    csa_inner_state_blocks: int = 4096,
) -> Path:
    kernel_dir = tmp_path / f"deepseek-v4-kernels-tp{lm_head_tp_size}"
    kernel_dir.mkdir()
    (kernel_dir / "prefill_attention_hca.py").write_text(
        "\n".join(
            [
                "HCA_STATE_BLOCK_NUM = 64",
                f"HCA_STATE_MAX_BLOCKS = {hca_state_blocks}",
                "",
            ]
        )
    )
    (kernel_dir / "prefill_attention_csa.py").write_text(
        "\n".join(
            [
                "CSA_STATE_BLOCK_NUM = 65",
                f"CSA_STATE_MAX_BLOCKS = {csa_state_blocks}",
                "INNER_STATE_BLOCK_NUM = 65",
                f"INNER_STATE_MAX_BLOCKS = {csa_inner_state_blocks}",
                "",
            ]
        )
    )
    (kernel_dir / "prefill_layer.py").write_text("")
    (kernel_dir / "prefill_fwd.py").write_text("")
    (kernel_dir / "prefill_mtp.py").write_text("")
    (kernel_dir / "decode_layer.py").write_text("")
    (kernel_dir / "decode_fwd.py").write_text("")
    (kernel_dir / "decode_mtp.py").write_text("")
    (kernel_dir / "config.py").write_text(
        "\n".join(
            [
                f"BLOCK_SIZE = {block_size}",
                "DECODE_BATCH = 4",
                "DECODE_SEQ = 2",
                "DECODE_TOKENS = DECODE_BATCH * DECODE_SEQ",
                "PREFILL_BATCH = 1",
                "PREFILL_SEQ = 128",
                "KV_ORI_MAX_BLOCKS = 128",
                "KV_ORI_TABLE_MAX_BLOCKS = 128",
                "KV_CMP_MAX_BLOCKS = 32",
                "IDX_CACHE_MAX_BLOCKS = 64",
                "ORI_KV_BLOCK_NUM = 128",
                "HCA_STATE_PHYSICAL_BLOCKS = 64",
                "CSA_STATE_PHYSICAL_BLOCKS = 65",
                "CSA_INNER_STATE_PHYSICAL_BLOCKS = 65",
                "PREFILL_ORI_MAX_BLOCKS = 128",
                "PREFILL_CMP_MAX_BLOCKS = KV_CMP_MAX_BLOCKS",
                "PREFILL_IDX_MAX_BLOCKS = IDX_CACHE_MAX_BLOCKS",
                "EP_WORLD_SIZE = 8",
                f"LM_HEAD_TP_SIZE = {lm_head_tp_size}",
                "",
            ]
        )
    )
    if use_config_constant:
        (kernel_dir / "lm_head.py").write_text("TP_SIZE = LM_HEAD_TP_SIZE\n")
    else:
        (kernel_dir / "lm_head.py").write_text(f"TP_SIZE = {lm_head_tp_size}\n")
    return kernel_dir


def _synthetic_layer_raw(*, layer_id: int, n_experts: int) -> dict[str, torch.Tensor]:
    prefix = f"layers.{layer_id}"
    raw = {
        f"{prefix}.hc_attn_fn": torch.arange(4, dtype=torch.float32).reshape(1, 4),
        f"{prefix}.hc_attn_scale": torch.arange(3, dtype=torch.float32),
        f"{prefix}.hc_attn_base": torch.arange(1, dtype=torch.float32),
        f"{prefix}.attn_norm.weight": torch.arange(4, dtype=torch.bfloat16),
        f"{prefix}.attn.wq_a.weight": torch.arange(8, dtype=torch.bfloat16).reshape(2, 4),
        f"{prefix}.attn.wq_b.weight": torch.arange(12, dtype=torch.int8).reshape(6, 2),
        f"{prefix}.attn.wq_b.scale": torch.arange(6, dtype=torch.float32),
        f"{prefix}.attn.wkv.weight": torch.arange(12, dtype=torch.bfloat16).reshape(3, 4),
        f"{prefix}.attn.q_norm.weight": torch.arange(2, dtype=torch.bfloat16),
        f"{prefix}.attn.kv_norm.weight": torch.arange(3, dtype=torch.bfloat16),
        f"{prefix}.attn.attn_sink": torch.arange(2, dtype=torch.float32),
        f"{prefix}.attn.wo_a.weight": torch.arange(64, dtype=torch.bfloat16).reshape(16, 4),
        f"{prefix}.attn.wo_b.weight": torch.arange(64, dtype=torch.int8).reshape(4, 16),
        f"{prefix}.attn.wo_b.scale": torch.arange(4, dtype=torch.float32),
        f"{prefix}.hc_ffn_fn": torch.arange(4, dtype=torch.float32).reshape(1, 4),
        f"{prefix}.hc_ffn_scale": torch.arange(3, dtype=torch.float32),
        f"{prefix}.hc_ffn_base": torch.arange(1, dtype=torch.float32),
        f"{prefix}.ffn_norm.weight": torch.arange(4, dtype=torch.bfloat16),
        f"{prefix}.ffn.gate.weight": torch.arange(16, dtype=torch.bfloat16).reshape(4, 4),
        f"{prefix}.ffn.gate.bias": torch.arange(4, dtype=torch.float32),
        f"{prefix}.ffn.shared_experts.w1.weight": torch.arange(8, dtype=torch.int8).reshape(2, 4),
        f"{prefix}.ffn.shared_experts.w1.scale": torch.arange(2, dtype=torch.float32),
        f"{prefix}.ffn.shared_experts.w2.weight": torch.arange(8, dtype=torch.int8).reshape(4, 2),
        f"{prefix}.ffn.shared_experts.w2.scale": torch.arange(4, dtype=torch.float32),
        f"{prefix}.ffn.shared_experts.w3.weight": torch.arange(8, dtype=torch.int8).reshape(2, 4),
        f"{prefix}.ffn.shared_experts.w3.scale": torch.arange(2, dtype=torch.float32),
        f"{prefix}.attn.compressor.wkv.weight": torch.arange(8, dtype=torch.bfloat16).reshape(2, 4),
        f"{prefix}.attn.compressor.wgate.weight": torch.arange(8, dtype=torch.bfloat16).reshape(2, 4),
        f"{prefix}.attn.compressor.ape": torch.arange(8, dtype=torch.float32).reshape(4, 2),
        f"{prefix}.attn.compressor.norm.weight": torch.arange(3, dtype=torch.bfloat16),
        f"{prefix}.attn.indexer.wq_b.weight": torch.arange(12, dtype=torch.int8).reshape(6, 2),
        f"{prefix}.attn.indexer.wq_b.scale": torch.arange(6, dtype=torch.float32),
        f"{prefix}.attn.indexer.weights_proj.weight": torch.arange(8, dtype=torch.bfloat16).reshape(2, 4),
        f"{prefix}.attn.indexer.compressor.wkv.weight": torch.arange(8, dtype=torch.bfloat16).reshape(2, 4),
        f"{prefix}.attn.indexer.compressor.wgate.weight": torch.arange(8, dtype=torch.bfloat16).reshape(2, 4),
        f"{prefix}.attn.indexer.compressor.ape": torch.arange(8, dtype=torch.float32).reshape(4, 2),
        f"{prefix}.attn.indexer.compressor.norm.weight": torch.arange(2, dtype=torch.bfloat16),
    }
    for expert_id in range(n_experts):
        base = expert_id * 10
        raw.update(
            {
                f"{prefix}.ffn.experts.{expert_id}.w1.weight": torch.full((2, 4), base, dtype=torch.int8),
                f"{prefix}.ffn.experts.{expert_id}.w1.scale": torch.full((2,), base + 1, dtype=torch.float32),
                f"{prefix}.ffn.experts.{expert_id}.w2.weight": torch.full((4, 2), base + 2, dtype=torch.int8),
                f"{prefix}.ffn.experts.{expert_id}.w2.scale": torch.full((4,), base + 3, dtype=torch.float32),
                f"{prefix}.ffn.experts.{expert_id}.w3.weight": torch.full((2, 4), base + 4, dtype=torch.int8),
                f"{prefix}.ffn.experts.{expert_id}.w3.scale": torch.full((2,), base + 5, dtype=torch.float32),
            }
        )
    return raw


class _Tokenizer:
    bos_token_id = 0
    eos_token_id = 1
    pad_token_id = None

    def encode(self, text: str) -> list[int]:
        return [1]

    def decode(self, token_ids: list[int]) -> str:
        return ""


def _runtime_model_for_embeddings():
    from pypto_serving.config.types import ModelConfig, RuntimeModel

    config = ModelConfig(
        model_id="dsv4",
        architecture="DeepseekV4ForCausalLM",
        vocab_size=6,
        hidden_size=4,
        intermediate_size=8,
        num_hidden_layers=43,
        num_attention_heads=64,
        num_key_value_heads=1,
        head_dim=512,
        max_position_embeddings=8192,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        bos_token_id=0,
        eos_token_id=1,
        pad_token_id=1,
        torch_dtype="bfloat16",
    )
    runtime = RuntimeConfig(page_size=128, max_batch_size=1, max_seq_len=260, weight_dtype="int8")
    placeholder = torch.empty(0, config.hidden_size)
    return RuntimeModel(
        config=config,
        runtime=runtime,
        embed_tokens=placeholder,
        final_norm_weight=torch.empty(0),
        lm_head=placeholder,
        layers=[],
    )


def _runner_for_prepared_inputs() -> tuple[DeepSeekV4ModelRunner, object]:
    model = _runtime_model_for_embeddings()
    compiled = DeepSeekV4CompiledKernels(
        # Exercise the production MTP tile while keeping the tiny host-side model.
        layout=DeepSeekV4CacheLayout(decode_batch=4, decode_seq=2, decode_tokens=8),
        model_dir="",
        weight_map={},
        weight_store=None,
        compress_ratios=tuple([0] * 44),
        layer_plan=build_deepseek_v4_layer_plan(
            compress_ratios=tuple([0] * 44),
            num_hidden_layers=43,
            num_hash_layers=3,
        ),
        kernel_dir="",
    )
    runner = DeepSeekV4ModelRunner(compiled=compiled)
    runner.init_kv_cache("dsv4", model.config, model.runtime)
    return runner, model


def test_deepseek_lm_head_computes_selected_rows_on_host_without_padded_vocab():
    layout = DeepSeekV4CacheLayout(ranks=2, decode_batch=2, decode_seq=2, decode_tokens=4)
    compiled = DeepSeekV4CompiledKernels(
        layout=layout,
        model_dir="",
        weight_map={},
        weight_store=None,
        compress_ratios=(),
        layer_plan=(),
        kernel_dir="",
    )
    runner = DeepSeekV4ModelRunner(compiled=compiled)
    lm_head_weight = torch.zeros((layout.ranks, 4, 3), dtype=torch.bfloat16)
    lm_head_weight[0, 0] = torch.tensor([1.0, 0.0, 0.0])
    lm_head_weight[0, 1] = torch.tensor([0.0, 1.0, 0.0])
    lm_head_weight[0, 2] = torch.tensor([0.0, 0.0, 1.0])
    lm_head_weight[1, 0] = torch.tensor([1.0, 1.0, 0.0])
    lm_head_weight[1, 1] = torch.tensor([0.0, 1.0, 1.0])
    runner._global_weights = weight_loader.DeepSeekV4GlobalWeights(
        embed_weight=torch.empty(0),
        final_norm_weight=torch.empty(0),
        lm_head_weight=lm_head_weight,
        lm_head_layout=weight_loader.DeepSeekV4LmHeadLayout(
            ranks=layout.ranks,
            vocab_size=5,
            hidden_size=3,
            vocab_per_rank=3,
            padded_vocab_per_rank=4,
        ),
        hc_head_fn=torch.empty(0),
        hc_head_scale=torch.empty(0),
        hc_head_base=torch.empty(0),
    )
    hidden = torch.arange(layout.ranks * 6 * 3, dtype=torch.float32).reshape(layout.ranks, 6, 3).to(torch.bfloat16)

    def fail_run_l3(*args):
        raise AssertionError("host LM-head must not dispatch an L3 program")

    runner._run_l3 = fail_run_l3
    logits = runner._logits_for_hidden(hidden, active_rows=(5, 2))

    assert logits.shape == (2, 5)
    assert logits[0].tolist() == [15, 16, 17, 31, 33]
    assert logits[1].tolist() == [6, 7, 8, 13, 15]


def test_deepseek_static_lm_head_weight_replicates_one_vocab_shard_per_dp_rank():
    layout = DeepSeekV4CacheLayout()
    tp_size = DEEPSEEK_V4_LM_HEAD_TP_SIZE
    # The regression only shows up when the DP world spans more than one TP group.
    assert layout.ranks > tp_size
    compiled = DeepSeekV4CompiledKernels(
        layout=layout,
        model_dir="",
        weight_map={},
        weight_store=None,
        compress_ratios=(),
        layer_plan=(),
        kernel_dir="",
    )
    runner = DeepSeekV4ModelRunner(compiled=compiled)
    vocab_per_rank = 2
    hidden_size = 3
    # Shard s carries the constant s + 1 so every rank's copy is identifiable.
    packed = torch.stack(
        [
            torch.full((vocab_per_rank, hidden_size), float(shard + 1), dtype=torch.bfloat16)
            for shard in range(tp_size)
        ]
    )
    runner._global_weights = weight_loader.DeepSeekV4GlobalWeights(
        embed_weight=torch.empty(0),
        final_norm_weight=torch.empty(0),
        lm_head_weight=packed,
        lm_head_layout=weight_loader.DeepSeekV4LmHeadLayout(
            ranks=tp_size,
            vocab_size=tp_size * vocab_per_rank,
            hidden_size=hidden_size,
            vocab_per_rank=vocab_per_rank,
            padded_vocab_per_rank=vocab_per_rank,
        ),
        hc_head_fn=torch.empty(0),
        hc_head_scale=torch.empty(0),
        hc_head_base=torch.empty(0),
    )

    replicated = runner._static_lm_head_weight_tensor()

    # Every card holds a full shard, not just the first TP group: ranks 4..7 must
    # repeat shards 0..3 instead of reading whatever the kernel maps past rank 3.
    assert replicated.shape == (layout.ranks, vocab_per_rank, hidden_size)
    for rank in range(layout.ranks):
        assert torch.equal(replicated[rank], packed[rank % tp_size]), f"rank {rank} shard mismatch"
    assert [float(replicated[rank, 0, 0]) for rank in range(layout.ranks)] == [
        1.0,
        2.0,
        3.0,
        4.0,
        1.0,
        2.0,
        3.0,
        4.0,
    ]


def test_deepseek_final_hidden_normalizes_before_hc_head_projection_overflows():
    compiled = DeepSeekV4CompiledKernels(
        layout=DeepSeekV4CacheLayout(),
        model_dir="",
        weight_map={},
        weight_store=None,
        compress_ratios=(),
        layer_plan=(),
        kernel_dir="",
    )
    runner = DeepSeekV4ModelRunner(compiled=compiled)
    hidden_size = 3
    runner._global_weights = weight_loader.DeepSeekV4GlobalWeights(
        embed_weight=torch.empty(0),
        final_norm_weight=torch.ones(hidden_size),
        lm_head_weight=torch.empty(0),
        lm_head_layout=weight_loader.DeepSeekV4LmHeadLayout(
            ranks=1,
            vocab_size=1,
            hidden_size=hidden_size,
            vocab_per_rank=1,
            padded_vocab_per_rank=1,
        ),
        hc_head_fn=torch.ones((4, hidden_size * 4), dtype=torch.float32),
        hc_head_scale=torch.ones((1,), dtype=torch.float32),
        hc_head_base=torch.zeros((4,), dtype=torch.float32),
    )
    x_hc = torch.full(
        (1, 2, 4, hidden_size),
        torch.finfo(torch.bfloat16).max,
        dtype=torch.bfloat16,
    )

    flat = x_hc.flatten(2).float()
    inv_rms = torch.rsqrt(flat.square().mean(dim=-1, keepdim=True) + 1e-6)
    unstable_mixes = torch.matmul(flat, runner._global_weights.hc_head_fn.t()) * inv_rms
    assert not torch.isfinite(unstable_mixes).all()

    hidden = runner._final_hidden(x_hc)

    assert hidden.shape == (1, 2, hidden_size)
    assert torch.isfinite(hidden.float()).all()
