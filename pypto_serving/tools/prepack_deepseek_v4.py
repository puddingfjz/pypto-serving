# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Build the optional prepacked DeepSeek V4 hidden-layer weight sidecar."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from pathlib import Path

from pypto_serving.model.deepseek.weight_loader import (
    DEEPSEEK_V4_PACKED_FORMAT,
    DeepSeekV4WeightStore,
    deepseek_v4_packed_weights_path,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Prepack DeepSeek V4 hidden-layer weights once so serving can mmap "
            "the final rank-stacked layout on later starts."
        )
    )
    parser.add_argument("model_dir", type=Path, help="DeepSeek V4 W8A8 checkpoint directory")
    parser.add_argument("--ranks", type=int, default=8, help="rank count for the packed layout")
    parser.add_argument(
        "--output",
        type=Path,
        help="output sidecar path; defaults to the serving auto-discovery path",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace an existing sidecar",
    )
    return parser


def _load_inputs(model_dir: Path) -> tuple[dict, dict[str, str]]:
    config_path = model_dir / "config.json"
    index_path = model_dir / "model.safetensors.index.json"
    config = json.loads(config_path.read_text())
    index = json.loads(index_path.read_text())
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not all(
        isinstance(name, str) and isinstance(filename, str)
        for name, filename in weight_map.items()
    ):
        raise ValueError(f"{index_path} must contain a string-to-string weight_map")
    return config, weight_map


def build_sidecar(
    model_dir: Path,
    *,
    ranks: int,
    output: Path,
    force: bool,
) -> None:
    """Pack and atomically publish one DeepSeek V4 sidecar."""
    if output.exists() and not force:
        raise FileExistsError(f"{output} already exists; pass --force to replace it")
    config, weight_map = _load_inputs(model_dir)
    num_hidden_layers = int(config["num_hidden_layers"])
    compress_ratios = tuple(int(value) for value in config["compress_ratios"][:num_hidden_layers])
    n_routed_experts = int(config["n_routed_experts"])
    num_hash_layers = int(config.get("num_hash_layers", 3))
    store = DeepSeekV4WeightStore(model_dir=model_dir, weight_map=weight_map)

    started = time.perf_counter()
    weights = store.load_stacked_layer_weights(
        ranks=ranks,
        n_routed_experts=n_routed_experts,
        compress_ratios=compress_ratios,
        num_hash_layers=num_hash_layers,
        use_prepacked=False,
    )
    fingerprint = store.packed_stacked_layer_weights_fingerprint(
        ranks=ranks,
        n_routed_experts=n_routed_experts,
        compress_ratios=compress_ratios,
        num_hash_layers=num_hash_layers,
    )

    try:
        from safetensors.torch import save_file
    except ImportError as exc:
        raise RuntimeError("safetensors is required to write packed DeepSeekV4 weights.") from exc

    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.",
        suffix=".tmp",
        dir=output.parent,
    )
    os.close(fd)
    temporary_path = Path(temporary_name)
    try:
        save_file(
            dict(weights.tensors),
            str(temporary_path),
            metadata={
                "format": DEEPSEEK_V4_PACKED_FORMAT,
                "source_fingerprint": fingerprint,
            },
        )
        os.replace(temporary_path, output)
    finally:
        temporary_path.unlink(missing_ok=True)

    size = output.stat().st_size
    elapsed = time.perf_counter() - started
    print(
        f"Wrote {output} ({size / 2**30:.3f} GiB) in {elapsed:.3f} seconds",
        flush=True,
    )


def main() -> int:
    """Run the prepack command."""
    args = build_parser().parse_args()
    model_dir = args.model_dir.resolve()
    output = (
        deepseek_v4_packed_weights_path(model_dir, ranks=args.ranks)
        if args.output is None
        else args.output.resolve()
    )
    build_sidecar(
        model_dir,
        ranks=args.ranks,
        output=output,
        force=args.force,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
