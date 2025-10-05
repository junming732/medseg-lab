from typing import Tuple, Optional
import torch
from torch import nn
import pytorch_lightning as pl
from monai.networks.nets import UNETR

class DiceBCELoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = self.bce(logits, targets)
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()
        intersect = (preds * targets).sum(dim=(1,2,3,4))
        dice = (2 * intersect + self.smooth) / (preds.sum(dim=(1,2,3,4)) + targets.sum(dim=(1,2,3,4)) + self.smooth)
        dice_loss = 1 - dice.mean()
        return bce + dice_loss

def binary_dice(preds: torch.Tensor, targets: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    preds = (preds > 0.5).float()
    inter = (preds * targets).sum(dim=(1,2,3,4))
    return ((2 * inter + eps) / (preds.sum(dim=(1,2,3,4)) + targets.sum(dim=(1,2,3,4)) + eps)).mean()

class UNETRLightning(pl.LightningModule):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        img_size: Tuple[int,int,int] = (32,32,32),
        feature_size: int = 16,
        learning_rate: float = 1e-3,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.net = UNETR(
            in_channels=in_channels,
            out_channels=out_channels,
            img_size=img_size,
            feature_size=feature_size,
            hidden_size=feature_size*24,
            mlp_dim=feature_size*48,
            num_heads=4,
            pos_embed="perceptron",
            norm_name="instance",
            conv_block=True,
            res_block=True,
            dropout_rate=0.0,
        )
        self.loss_fn = DiceBCELoss()
        self.learning_rate = learning_rate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = self.loss_fn(logits, y)
        with torch.no_grad():
            dice = binary_dice(torch.sigmoid(logits), y)
        self.log("train/loss", loss, prog_bar=True)
        self.log("train/dice", dice, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = self.loss_fn(logits, y)
        dice = binary_dice(torch.sigmoid(logits), y)
        self.log("val/loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log("val/dice", dice, prog_bar=True, on_epoch=True, on_step=False)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=1e-4)
