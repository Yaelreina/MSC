"""
PetriAdapter: fuses Transformer visual features with Petri net marking vectors.

Receives (B, T, D_vis) visual features and (B, T, num_places) marking vectors;
learns to handle "disagreement" when video evidence contradicts the net's legal next steps
(~0.79 fitness → controlled conflict for robustness).
"""

from __future__ import annotations

from typing import Optional, Tuple

try:
    import torch
    import torch.nn as nn
except ImportError:
    raise ImportError("petri_adapter requires torch") from None


class PetriAdapter(nn.Module):
    """
    Adapter that combines Transformer visual features with 32-dim (or num_places) marking vectors.

    Inputs:
        - visual_features: (B, T, D_vis) from ASFormer/Transformer
        - marking_vectors: (B, T, num_places) binary 0/1 from dynamic_marking
        - disagreement_mask: optional (B, T) bool, True where video contradicts net (for loss/conditioning)
    """

    def __init__(
        self,
        visual_dim: int,
        num_places: int = 32,
        hidden_dim: int = 256,
        output_dim: Optional[int] = None,
        fusion: str = "concat",  # "concat" | "gate" | "marking_only"
        use_disagreement_conditioning: bool = False,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.visual_dim = visual_dim
        self.num_places = num_places
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim or visual_dim
        self.fusion = fusion
        self.use_disagreement_conditioning = use_disagreement_conditioning

        if fusion == "concat":
            self.marking_proj = nn.Linear(num_places, hidden_dim)
            self.vis_proj = nn.Linear(visual_dim, hidden_dim)
            self.fuse = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, self.output_dim),
            )
        elif fusion == "gate":
            self.marking_proj = nn.Linear(num_places, visual_dim)
            self.gate = nn.Sequential(
                nn.Linear(visual_dim + num_places, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, visual_dim),
                nn.Sigmoid(),
            )
            self.out_proj = nn.Linear(visual_dim, self.output_dim)
        elif fusion == "marking_only":
            self.marking_proj = nn.Sequential(
                nn.Linear(num_places, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, self.output_dim),
            )
        else:
            raise ValueError(f"fusion must be concat|gate|marking_only, got {fusion}")

        if use_disagreement_conditioning:
            # Optional: learn a modifier when there is disagreement (video vs net)
            self.disagreement_embed = nn.Parameter(torch.zeros(1, 1, self.output_dim))

        self._fusion = fusion

    def forward(
        self,
        visual_features: torch.Tensor,
        marking_vectors: torch.Tensor,
        disagreement_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            visual_features: (B, T, D_vis)
            marking_vectors: (B, T, num_places)
            disagreement_mask: (B, T) optional; True = frame has disagreement

        Returns:
            (B, T, output_dim) fused representation
        """
        B, T, _ = visual_features.shape
        assert marking_vectors.shape == (B, T, self.num_places), (
            f"marking_vectors shape {marking_vectors.shape} vs expected (B, T, {self.num_places})"
        )

        if self._fusion == "concat":
            vis_h = self.vis_proj(visual_features)
            mark_h = self.marking_proj(marking_vectors)
            fused = self.fuse(torch.cat([vis_h, mark_h], dim=-1))
        elif self._fusion == "gate":
            gate = self.gate(torch.cat([visual_features, marking_vectors], dim=-1))
            marked = self.marking_proj(marking_vectors)
            fused = self.out_proj(gate * visual_features + (1 - gate) * marked)
        else:
            fused = self.marking_proj(marking_vectors)

        if self.use_disagreement_conditioning and disagreement_mask is not None:
            # Add learned embedding where there is disagreement (controlled conflict)
            disp = self.disagreement_embed.expand(B, T, -1)
            fused = fused + disp * disagreement_mask.unsqueeze(-1).to(fused.dtype)

        return fused


def disagreement_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    disagreement_mask: torch.Tensor,
    weight_disagreement: float = 1.0,
    ignore_index: int = -100,
) -> torch.Tensor:
    """
    Cross-entropy loss with optional extra weight on frames where video and net disagree.

    Use this to train the adapter to be robust when the process model says "illegal" but
    the video shows the action (fitness ~0.79 → controlled conflict).

    Args:
        logits: (B, T, num_classes) or (B*T, num_classes)
        targets: (B, T) class indices; ignore_index entries are ignored
        disagreement_mask: (B, T) True where we have disagreement
        weight_disagreement: multiplier for loss on disagreement frames
        ignore_index: target value to ignore

    Returns:
        scalar loss
    """
    if logits.dim() == 3:
        B, T, C = logits.shape
        logits = logits.view(B * T, C)
        targets = targets.view(B * T)
        disagreement_mask = disagreement_mask.view(B * T)
    else:
        B_T = logits.shape[0]
        disagreement_mask = disagreement_mask.view(B_T)

    ce = nn.functional.cross_entropy(
        logits, targets, reduction="none", ignore_index=ignore_index
    )
    valid = targets != ignore_index
    if weight_disagreement != 1.0 and disagreement_mask.any():
        disagree = disagreement_mask.bool() if disagreement_mask.is_floating_point() else disagreement_mask
        weight = torch.ones_like(ce, device=ce.device)
        weight[valid & disagree] = weight_disagreement
        ce = ce * weight
    return ce[valid].mean()
