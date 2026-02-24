from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import re

import torch
import torch.nn as nn


@dataclass
class LoadReport:
    loaded: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    unexpected: list[str] = field(default_factory=list)
    mismatched: list[str] = field(default_factory=list)


def _load_single_weight_file(path: Path) -> dict[str, torch.Tensor]:
    if path.suffix == ".safetensors":
        try:
            from safetensors.torch import load_file
        except ImportError as exc:
            raise ImportError(
                "safetensors is required to load .safetensors files. "
                "Install with: pip install safetensors"
            ) from exc
        return load_file(str(path))
    return torch.load(path, map_location="cpu")


def _load_from_index(index_path: Path) -> dict[str, torch.Tensor]:
    index = json.loads(index_path.read_text())
    weight_map: dict[str, str] = index.get("weight_map", {})
    if not weight_map:
        raise ValueError(f"Invalid index file: {index_path}")

    state_dict: dict[str, torch.Tensor] = {}
    for shard_name in sorted(set(weight_map.values())):
        shard_path = index_path.parent / shard_name
        shard_state = _load_single_weight_file(shard_path)
        state_dict.update(shard_state)
    return state_dict


def load_hf_state_dict(model_path: str | Path) -> dict[str, torch.Tensor]:
    """
    Load a HF style state_dict from a local directory or a single weight file.
    Supported files:
    - model.safetensors / model.safetensors.index.json
    - pytorch_model.bin / pytorch_model.bin.index.json
    """
    path = Path(model_path)
    if path.is_file():
        return _load_single_weight_file(path)

    if not path.exists():
        raise FileNotFoundError(f"Path not found: {path}")

    candidates = [
        path / "model.safetensors",
        path / "pytorch_model.bin",
    ]
    for weight_file in candidates:
        if weight_file.exists():
            return _load_single_weight_file(weight_file)

    index_candidates = [
        path / "model.safetensors.index.json",
        path / "pytorch_model.bin.index.json",
    ]
    for index_file in index_candidates:
        if index_file.exists():
            return _load_from_index(index_file)

    raise FileNotFoundError(
        f"No supported weight file found under: {path}. "
        "Expected model.safetensors, pytorch_model.bin, "
        "or their index json."
    )


def _infer_num_layers_from_hf_state(hf_state_dict: dict[str, torch.Tensor]) -> int:
    pattern = re.compile(r"^model\.layers\.(\d+)\.")
    max_idx = -1
    for key in hf_state_dict:
        match = pattern.match(key)
        if match:
            max_idx = max(max_idx, int(match.group(1)))
    return max_idx + 1


def remap_hf_qwen3_state_dict(
    hf_state_dict: dict[str, torch.Tensor],
    num_layers: int,
) -> dict[str, torch.Tensor]:
    """
    Remap HF Qwen3 parameter names to current T-LLM model names.
    """
    mapped: dict[str, torch.Tensor] = {}

    def put(dst: str, src: str):
        tensor = hf_state_dict.get(src)
        if tensor is not None:
            mapped[dst] = tensor

    put("model.embed_tokens.weight", "model.embed_tokens.weight")
    put("model.norm.weight", "model.norm.weight")

    if "lm_head.weight" in hf_state_dict:
        mapped["lm_head.weight"] = hf_state_dict["lm_head.weight"]
    elif "model.embed_tokens.weight" in hf_state_dict:
        # fallback for tied embedding models that do not store lm_head.
        mapped["lm_head.weight"] = hf_state_dict["model.embed_tokens.weight"]

    for i in range(num_layers):
        layer_hf = f"model.layers.{i}"
        layer_tllm = f"model.layers.{i}"

        put(
            f"{layer_tllm}.input_layernorm.weight",
            f"{layer_hf}.input_layernorm.weight",
        )
        put(
            f"{layer_tllm}.post_attention_layernorm.weight",
            f"{layer_hf}.post_attention_layernorm.weight",
        )
        put(
            f"{layer_tllm}.self_attn.o_proj.weight",
            f"{layer_hf}.self_attn.o_proj.weight",
        )
        put(
            f"{layer_tllm}.mlp.gate_down.weight",
            f"{layer_hf}.mlp.down_proj.weight",
        )

        q_w = hf_state_dict.get(f"{layer_hf}.self_attn.q_proj.weight")
        k_w = hf_state_dict.get(f"{layer_hf}.self_attn.k_proj.weight")
        v_w = hf_state_dict.get(f"{layer_hf}.self_attn.v_proj.weight")
        if q_w is not None and k_w is not None and v_w is not None:
            mapped[f"{layer_tllm}.self_attn.qkv_proj.weight"] = torch.cat(
                [q_w, k_w, v_w], dim=0
            )

        gate_w = hf_state_dict.get(f"{layer_hf}.mlp.gate_proj.weight")
        up_w = hf_state_dict.get(f"{layer_hf}.mlp.up_proj.weight")
        if gate_w is not None and up_w is not None:
            mapped[f"{layer_tllm}.mlp.gate_up.weight"] = torch.cat(
                [gate_w, up_w], dim=0
            )

    return mapped


@torch.no_grad()
def _copy_param_or_buffer(dst: torch.Tensor, src: torch.Tensor) -> None:
    src = src.to(device=dst.device, dtype=dst.dtype)
    if dst.shape != src.shape:
        raise ValueError(f"shape mismatch: model={tuple(dst.shape)} hf={tuple(src.shape)}")
    dst.copy_(src)


@torch.no_grad()
def load_mapped_state_dict(
    model: nn.Module,
    mapped_state_dict: dict[str, torch.Tensor],
    strict: bool = False,
) -> LoadReport:
    """
    Load already-remapped tensors into model and return a detailed report.
    """
    report = LoadReport()
    params = dict(model.named_parameters())
    buffers = dict(model.named_buffers())
    loaded_names: set[str] = set()

    for name, src in mapped_state_dict.items():
        if name in params:
            try:
                dst = params[name]
                weight_loader = getattr(dst, "weight_loader", None)
                if callable(weight_loader):
                    weight_loader(dst, src)
                else:
                    _copy_param_or_buffer(dst, src)
                report.loaded.append(name)
                loaded_names.add(name)
            except Exception as exc:
                report.mismatched.append(f"{name}: {exc}")
        elif name in buffers:
            try:
                _copy_param_or_buffer(buffers[name], src)
                report.loaded.append(name)
                loaded_names.add(name)
            except Exception as exc:
                report.mismatched.append(f"{name}: {exc}")
        else:
            report.unexpected.append(name)

    for name in params:
        if name not in loaded_names:
            report.missing.append(name)
    for name in buffers:
        if name not in loaded_names and name in mapped_state_dict:
            report.missing.append(name)

    if strict and (report.missing or report.unexpected or report.mismatched):
        raise RuntimeError(
            "Weight loading failed in strict mode. "
            f"missing={len(report.missing)}, "
            f"unexpected={len(report.unexpected)}, "
            f"mismatched={len(report.mismatched)}"
        )
    return report


def load_qwen3_weights(
    model: nn.Module,
    source: str | Path | dict[str, torch.Tensor],
    strict: bool = False,
) -> LoadReport:
    """
    Minimal Qwen3 weight loader.

    Usage:
    - load_qwen3_weights(model, "/path/to/hf_model_dir")
    - load_qwen3_weights(model, hf_state_dict)
    """
    if isinstance(source, dict):
        hf_state_dict = source
    else:
        hf_state_dict = load_hf_state_dict(source)

    if hasattr(model, "model") and hasattr(model.model, "layers"):
        num_layers = len(model.model.layers)
    else:
        num_layers = _infer_num_layers_from_hf_state(hf_state_dict)

    mapped = remap_hf_qwen3_state_dict(hf_state_dict, num_layers=num_layers)
    return load_mapped_state_dict(model, mapped, strict=strict)
