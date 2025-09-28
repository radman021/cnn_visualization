from pathlib import Path
import uuid

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from logger import Logger


class LayerOutputVisualizer:

    def __init__(self, model, device: str = None):
        self.model = model.to(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.layer_outputs = {}
        self.hooks = []
        self.conv_order = []
        self.logger = Logger("image_propagation").get_logger()

    def _hook_fn(self, name):
        def hook(module, input, output):
            self.layer_outputs[name] = output.detach().cpu()

        return hook

    def register_hooks(self):
        self.remove_hooks()
        self.conv_order.clear()
        for name, layer in self.model.named_modules():
            if isinstance(layer, nn.Conv2d):
                self.conv_order.append((name, layer))
                self.hooks.append(layer.register_forward_hook(self._hook_fn(name)))

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def save_layer_outputs(
        self,
        input_tensor: torch.Tensor,
        base_storage: str,
        image_name: str | None = None,
        normalize_each: bool = True,
        cmap: str = "gray",
        downsample_to: int | None = 128,
    ):
        self.layer_outputs.clear()
        with torch.no_grad():
            _ = self.model(input_tensor.to(self.device))

        if image_name:
            run_dir = Path(base_storage) / image_name
        else:
            run_dir = Path(base_storage) / f"image{uuid.uuid4().hex}"

        self.logger.info(f"Started layer outputs saving to folder: {run_dir}")
        run_dir.mkdir(parents=True, exist_ok=True)

        def _downsample(t: torch.Tensor) -> torch.Tensor:
            if downsample_to is None:
                return t
            _, _, H, W = t.shape
            if max(H, W) <= downsample_to:
                return t
            if H >= W:
                new_h = downsample_to
                new_w = max(1, int(round(W * (downsample_to / H))))
            else:
                new_w = downsample_to
                new_h = max(1, int(round(H * (downsample_to / W))))
            return F.interpolate(
                t, size=(new_h, new_w), mode="bilinear", align_corners=False
            )

        def _to_numpy(img_t: torch.Tensor) -> np.ndarray:
            arr = img_t.detach().cpu().float().numpy()
            if normalize_each:
                mn, mx = arr.min(), arr.max()
                arr = (arr - mn) / (mx - mn) if mx > mn else np.zeros_like(arr)
            return arr

        for name, layer in self.conv_order:
            if name not in self.layer_outputs:
                continue
            fmap = self.layer_outputs[name]
            fmap = _downsample(fmap)

            if name.startswith("conv1"):
                stage_name = "Block1"
                conv_name = "conv1"
            elif name.startswith("layer"):
                parts = name.split(".")
                stage_num = int(parts[0][5:])
                stage_name = f"Block{stage_num + 1}"
                conv_name = "_".join(parts)
            else:
                stage_name = "BlockX"
                conv_name = name.replace(".", "_")

            conv_dir = run_dir / stage_name / conv_name
            conv_dir.mkdir(parents=True, exist_ok=True)

            C = fmap.shape[1]
            self.logger.info(f"Saving layer outputs from layer {name} to {conv_dir}")

            for c in range(C):
                img = _to_numpy(fmap[0, c])
                plt.imsave(conv_dir / f"output{c}.png", img, cmap=cmap)

        self.logger.info(f"Successfully saved all layer outputs to {run_dir}")
        return run_dir
