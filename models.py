"""
Hybrid DINO–ViT–MoE architecture for Alzheimer's Disease Classification.

Components
----------
- Self-supervised DINO Vision Transformer (ViT-B/16)
- ImageNet-pretrained Vision Transformer (ViT-B/16)
- Feature Fusion Layer
- Mixture-of-Experts (MoE) Classification Head
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import timm
from torchvision import models

class LabelSmoothingCrossEntropy(nn.Module):
    """Cross-entropy loss with label smoothing."""

    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:

        target = target.to(pred.device).long()
        confidence = 1.0 - self.smoothing
        log_probs = F.log_softmax(pred, dim=-1)
        nll_loss = -log_probs.gather(
            dim=-1,
            index=target.unsqueeze(1)
        ).squeeze(1)
        smooth_loss = -log_probs.mean(dim=-1)
        loss = confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

# ---------- MoE head ----------
class MoEHead(nn.Module):
    """Mixture-of-Experts classification head."""

    def __init__(self, input_dim: int, num_classes: int,
        num_experts: int = 3, hidden_dim: int = 512, dropout: float = 0.5,):
        
        super().__init__()
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes),
            )
            for _ in range(num_experts)])

        self.gate = nn.Linear(input_dim, num_experts)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        gate_weights = self.softmax(self.gate(x))
        expert_outputs = torch.stack([expert(x) for expert in self.experts],dim=2)
        output = (expert_outputs * gate_weights.unsqueeze(1)).sum(dim=2)

        return output
    
# ---------- DINO wrapper ----------
class DINOWrapper(nn.Module):
    """
    Wrapper around timm DINO model.
    - By default we'll freeze everything initially inside this wrapper,
      then caller can call unfreeze_last_n_blocks(n) to unfreeze last n blocks.
    - encode_image does NOT use @torch.no_grad so gradient can flow for unfrozen parts.
    """
    def __init__(self, model_name="vit_base_patch16_224_dino", pretrained=True, device="cpu"):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)
        self.model.to(device)

        # freeze by default; unfreeze_last_n_blocks will flip for last blocks
        for p in self.model.parameters():
            p.requires_grad = False
            
        self._device = next(self.model.parameters()).device
        # infer output dim robustly
        self.output_dim = (
            getattr(self.model, "embed_dim", None)
            or getattr(self.model, "num_features", None)
            or 768
        )

    def encode_image(self, x):
        """
        x: tensor on caller device (B,C,H,W)
        returns: (B, D) CLS or pooled features (no device change requirements from caller)
        """
        x = x.to(self._device)
        if hasattr(self.model, "forward_features"):
            feats = self.model.forward_features(x)
            if feats is None:
                out = self.model(x)
                return out.view(out.size(0), -1)
            if feats.ndim == 2:
                return feats
            elif feats.ndim == 3:
                return feats[:, 0, :]
            else:
                return feats.view(feats.size(0), -1)
        else:
            out = self.model(x)
            if out.ndim == 2:
                return out
            else:
                return out.view(out.size(0), -1)

    def unfreeze_last_n_blocks(self, n):
        """
        Freeze all parameters then unfreeze last n transformer blocks.
        Works for many timm ViT-like models exposing .blocks or .encoder.layers
        """
        # already frozen in __init__, but ensure consistency
        for p in self.model.parameters():
            p.requires_grad = False

        # candidate attribute paths where blocks might be
        candidate_paths = [
            ["blocks"],
            ["transformer", "blocks"],
            ["blocks"],  # fallback repeats harmless
        ]
        blocks = None
        for path in candidate_paths:
            obj = self.model
            ok = True
            for attr in path:
                if hasattr(obj, attr):
                    obj = getattr(obj, attr)
                else:
                    ok = False
                    break
            if ok:
                blocks = obj
                break

        # other fallbacks
        if blocks is None:
            if hasattr(self.model, "blocks"):
                blocks = getattr(self.model, "blocks")
            elif hasattr(self.model, "encoder") and hasattr(self.model.encoder, "layers"):
                blocks = getattr(self.model.encoder, "layers")

        if blocks is None:
            # last resort: find child modules whose name contains "block" or "transformer"
            children = list(self.model.named_children())
            block_names = [name
                for name, _ in children
                if ("block" in name.lower()
                    or "transformer" in name.lower())]
            if block_names:
                last_name = block_names[-1]
                module = dict(children)[last_name]
                for p in module.parameters():
                    p.requires_grad = True
                print(f"[DINOWrapper] Warning: couldn't find blocks list; unfroze module {last_name}")
                return

            print("[DINOWrapper] Warning: couldn't locate transformer blocks to unfreeze.")
            return

        # indexable sequence expected
        try:
            total = len(blocks)
        except Exception:
            print("[DINOWrapper] Warning: transformer blocks object not indexable.")
            return

        n_unfreeze = min(max(0, int(n)), total)
        for blk in list(blocks)[-n_unfreeze:]:
            for p in blk.parameters():
                p.requires_grad = True

        print(f"[DINOWrapper] Unfroze last {n_unfreeze}/{total} blocks of DINO model.")

# ---------- Model ----------
class DINO_ViT_MoE(nn.Module):
    
    def __init__(self, dino_model_wrapper, num_classes, num_experts=3, finetune_vit_layers=3):
        super().__init__()
        self.dino = dino_model_wrapper  # wrapper already controls freezing/unfreezing

        # A separate ViT from torchvision to combine complementary features
        # we will freeze most of it and only unfreeze last finetune_vit_layers
        self.vit = models.vit_b_16(weights="IMAGENET1K_V1")
        vit_dim = self.vit.heads.head.in_features
        self.vit.heads = nn.Identity()

        for param in self.vit.parameters():
            param.requires_grad = False
            
        if finetune_vit_layers > 0:
            n_layers = len(self.vit.encoder.layers)
            n_unfreeze = min(finetune_vit_layers, n_layers)
            for layer in list(self.vit.encoder.layers)[-n_unfreeze:]:
                for p in layer.parameters():
                    p.requires_grad = True

        # get dino dim
        dino_dim = getattr(self.dino, "output_dim", 768)

        # fuse DINO and ViT features
        fused_dim = vit_dim  # project into vit_dim
        self.fc_proj = nn.Linear(dino_dim + vit_dim, fused_dim)
        self.moe_head = MoEHead(fused_dim, num_classes, num_experts=num_experts)

    def forward(self, x):
        # call dino without no_grad so unfrozen parts can receive gradients
        dino_feat = self.dino.encode_image(x)  # (B, D_dino)
        vit_feat = self.vit(x)                 # (B, D_vit)
        # ensure same device/dtype
        dino_feat = dino_feat.to(vit_feat.device).type(vit_feat.dtype)
        combined = torch.cat([dino_feat, vit_feat], dim=1)
        fused = self.fc_proj(combined)
        return self.moe_head(fused)

__all__ = [
    "LabelSmoothingCrossEntropy",
    "MoEHead",
    "DINOWrapper",
    "DINO_ViT_MoE"
]