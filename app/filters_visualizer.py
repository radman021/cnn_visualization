from typing import List, Tuple
import os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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
        base_dir: str,
        normalize_each: bool = True,
        cmap: str = "gray",
    ):
        self.logger.info(f"Started filter weights saving to folder: {base_dir}")
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

        self.logger.info(f"Successfully saved all filters.")

    def _compose_kernels_ignore_stride_padding(
        self, W_upper: torch.Tensor, W_lower_equiv: torch.Tensor
    ) -> torch.Tensor:
        """
        Compose upper-layer kernels with already 'lifted' lower kernels (cross-correlation form).
        Ignores stride/padding/dilation intentionally for a clean visualization.
        """
        outU, inU, kHu, kWu = W_upper.shape
        inU2, in0, kHl, kWl = W_lower_equiv.shape
        if inU != inU2:
            raise ValueError("Channel mismatch between composed layers.")

        kH = kHu + kHl - 1
        kW = kWu + kWl - 1
        K_equiv = torch.zeros((outU, in0, kH, kW), dtype=W_upper.dtype)

        for ou in range(outU):
            for i0 in range(in0):
                acc = None
                for iu in range(inU):
                    a = W_upper[ou, iu].unsqueeze(0).unsqueeze(0)
                    b = W_lower_equiv[iu, i0].unsqueeze(0).unsqueeze(0)
                    cur = F.conv2d(b, a, padding=(kHu - 1, kWu - 1))
                    cur = cur.squeeze(0).squeeze(0)
                    acc = cur if acc is None else acc + cur
                K_equiv[ou, i0] = acc
        return K_equiv

    def _equivalent_to_input_until(self, target_idx_1_based: int) -> torch.Tensor:
        """
        Return equivalent kernels of conv layer (1-based index in self.get_conv_layers())
        lifted to the INPUT space. Ignores stride/padding/dilation for visualization.
        Output: (outC_target, in0, kH_eq, kW_eq)
        """
        convs = self.get_conv_layers()
        if not (1 <= target_idx_1_based <= len(convs)):
            raise ValueError("Invalid conv index")

        _, base = convs[0]
        W_equiv = base.weight.detach().cpu()
        if target_idx_1_based == 1:
            return W_equiv

        for idx in range(2, target_idx_1_based + 1):
            _, layer = convs[idx - 1]
            W_upper = layer.weight.detach().cpu()
            W_equiv = self._compose_kernels_ignore_stride_padding(W_upper, W_equiv)
        return W_equiv

    def save_all_filters_linear_combinations(
        self,
        base_dir: str,
        normalize_each: bool = True,
        cmap: str = "gray",
    ):
        """
        For each Conv2d layer, compute its filters lifted to input (RGB) space by composing
        all previous convs. Save ONE aggregated image per output channel (L2 over input channels).
        """
        self.logger.info(f"Started linear-combination saving to folder: {base_dir}")
        os.makedirs(base_dir, exist_ok=True)
        convs = self.get_conv_layers()
        self.logger.info("Retrieved convolutional layers.")

        for idx, (name, layer) in enumerate(convs, start=1):
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

            try:
                K_eq = self._equivalent_to_input_until(idx)
            except ValueError as e:
                self.logger.warning(f"Skipping {name} due to composition mismatch: {e}")
                continue

            agg = torch.sqrt(torch.clamp((K_eq**2).sum(dim=1), 1e-12))
            outC = agg.shape[0]
            self.logger.info(
                f"Saving lifted filters for {name} -> {conv_dir}, shape={tuple(K_eq.shape)}"
            )

            for oc in range(outC):
                img = agg[oc].numpy()
                if normalize_each:
                    mn, mx = img.min(), img.max()
                    img = (img - mn) / (mx - mn) if mx > mn else np.zeros_like(img)
                plt.imsave(conv_dir / f"filter{oc}.png", img, cmap=cmap)

        self.logger.info("Successfully saved all lifted (linear-combination) filters.")
