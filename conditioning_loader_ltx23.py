import hashlib
from pathlib import Path
from typing import Any

import folder_paths
import safetensors
import torch
import json
from comfy_api.latest import io

from .nodes_registry import comfy_node


@comfy_node(name="LTXV23LoadConditioning")
class LTXV23LoadConditioning(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        files = folder_paths.get_filename_list("embeddings")
        if not files:
            files = [""]
        return io.Schema(
            node_id="LTXV23LoadConditioning",
            display_name="🅛🅣🅧 LTXV 2.3 Load Conditioning",
            category="lightricks/LTXV",
            inputs=[
                io.Combo.Input("file_name", options=sorted(files)),
                io.Combo.Input("device", options=["cpu", "gpu"]),
            ],
            outputs=[
                io.Conditioning.Output(),
            ],
        )


    @classmethod
    def execute(cls, file_name: str, device: str) -> io.NodeOutput:
        file_path = folder_paths.get_full_path("embeddings", file_name)
        if not Path(file_path).exists():
            raise FileNotFoundError(f"Conditioning file not found: {file_path}")

        target_device = "cpu"
        if device == "gpu":
            target_device = "cuda" if torch.cuda.is_available() else "cpu"

        conditioning: list[list[Any]] = []

        with safetensors.safe_open(file_path, framework="pt", device=target_device) as f:
            # restore options_meta from metadata
            meta = f.metadata() or {}
            options_meta = json.loads(meta.get("options_meta", "{}"))

            all_keys = list(f.keys())
            tensor_keys = sorted([k for k in all_keys if k.startswith("conditioning_data_")])

            for tensor_key in tensor_keys:
                idx = tensor_key.replace("conditioning_data_", "")
                tensor = f.get_tensor(tensor_key)

                options: dict[str, Any] = {}

                # restore tensor options (attention_mask, pooled_output, etc.)
                for k in all_keys:
                    prefix = f"option_{idx}_"
                    if k.startswith(prefix):
                        opt_name = k[len(prefix):]
                        options[opt_name] = f.get_tensor(k)

                # restore non-tensor options
                if idx in options_meta:
                    options.update(options_meta[idx])

                conditioning.append([tensor, options])

        if not conditioning:
            raise ValueError(f"No conditioning data found in file: {file_name}")

        return io.NodeOutput(conditioning)