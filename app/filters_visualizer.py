from typing import List, Tuple
import os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from logger import Logger


class FiltersVisualizer:

    def __init__(self, model: nn.Module, device: str | None = None):
        self.model = model
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.conv_order: List[Tuple[str, nn.Conv2d]] = []
        self._collect_convs()
        self.logger = Logger("filter_weights").get_logger()

    def _collect_convs(self) -> None:
        self.conv_order.clear()
        for name, layer in self.model.named_modules():
            if isinstance(layer, nn.Conv2d):
                self.conv_order.append((name, layer))

    def get_conv_layers(self) -> List[Tuple[str, nn.Conv2d]]:
        if not self.conv_order:
            self._collect_convs()
        return self.conv_order

    def save_all_filters_weights(
        self,
        base_dir,
        normalize_each: bool = True,
        cmap: str = "gray",
    ):
        self.logger.info(f"Started filter weidghts saving to folder: {base_dir}")
        os.makedirs(base_dir, exist_ok=True)
        convs = self.get_conv_layers()
        self.logger.info(f"Retrieved convolutional layers.")

        for name, layer in convs:
            self.logger.info(f"Saving filters from layer: {layer}")
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

            conv_dir = Path(base_dir) / stage_name / conv_name
            conv_dir.mkdir(parents=True, exist_ok=True)

            W = layer.weight.detach().cpu()
            outC, _, _, _ = W.shape
            agg = torch.sqrt(torch.clamp((W**2).sum(dim=1), 1e-12))

            self.logger.info(f"Saving filters to layer: {conv_dir}")

            for oc in range(outC):
                img = agg[oc].numpy()
                if normalize_each:
                    mn, mx = img.min(), img.max()
                    img = (img - mn) / (mx - mn) if mx > mn else np.zeros_like(img)
                plt.imsave(conv_dir / f"filter{oc}.png", img, cmap=cmap)

        self.logger.info(f"Sucessfullt saved all filters!")
