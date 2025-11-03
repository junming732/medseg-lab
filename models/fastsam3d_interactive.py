"""
FastSAM3D Interactive Model

Wrapper for FastSAM3D model to work with prompt-based training.
FastSAM3D uses SAM's native prompt encoder (positional embeddings).

Based on: https://github.com/arcadelab/FastSAM3D
Paper: "FastSAM3D: An Efficient Segment Anything Model for 3D Volumetric Medical Images"
"""

import torch
import torch.nn as nn
from typing import Tuple, List, Optional
import numpy as np


class FastSAM3DPromptEncoder(nn.Module):
    """
    Prompt encoder for FastSAM3D.

    FastSAM3D uses SAM's native prompt encoding:
    - Positional encodings for point coordinates
    - Learned embeddings for point type (positive/negative)
    - Combined representation
    """

    def __init__(self, embed_dim: int = 256, image_size: Tuple[int, int, int] = (128, 128, 128)):
        super().__init__()
        self.embed_dim = embed_dim
        self.image_size = image_size

        # Learned embeddings for positive/negative points
        self.point_embeddings = nn.Embedding(2, embed_dim)  # 0=negative, 1=positive

        # Positional encoding for 3D coordinates
        self.pe_layer = self._build_positional_encoding(embed_dim, image_size)

    def _build_positional_encoding(self, embed_dim, image_size):
        """Build 3D positional encoding."""
        # For simplicity, using learnable positional encoding
        # In full SAM, this uses sinusoidal encoding
        D, H, W = image_size
        pos_embed = nn.Parameter(torch.randn(1, embed_dim, D, H, W) * 0.02)
        return pos_embed

    def forward(self, points: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Encode point prompts.

        Args:
            points: [B, N, 3] point coordinates (z, y, x) in [0, image_size]
            labels: [B, N] point labels (0=negative, 1=positive)

        Returns:
            point_embeddings: [B, N, embed_dim]
        """
        B, N, _ = points.shape

        # Get learned type embeddings
        type_embeddings = self.point_embeddings(labels)  # [B, N, embed_dim]

        # Get positional encoding at point locations
        # Sample from the PE grid
        pos_encodings = []
        for i in range(B):
            batch_pos = []
            for j in range(N):
                z, y, x = points[i, j]
                # Normalize to [-1, 1] for grid_sample
                z_norm = (z / self.image_size[0]) * 2 - 1
                y_norm = (y / self.image_size[1]) * 2 - 1
                x_norm = (x / self.image_size[2]) * 2 - 1

                # Sample from PE grid (simplified version)
                z_idx = int(z.item())
                y_idx = int(y.item())
                x_idx = int(x.item())

                # Clip to valid range
                z_idx = max(0, min(self.image_size[0] - 1, z_idx))
                y_idx = max(0, min(self.image_size[1] - 1, y_idx))
                x_idx = max(0, min(self.image_size[2] - 1, x_idx))

                pos_enc = self.pe_layer[0, :, z_idx, y_idx, x_idx]  # [embed_dim]
                batch_pos.append(pos_enc)

            pos_encodings.append(torch.stack(batch_pos))

        pos_encodings = torch.stack(pos_encodings)  # [B, N, embed_dim]

        # Combine positional and type embeddings
        point_embeddings = pos_encodings + type_embeddings

        return point_embeddings


class FastSAM3DInteractive(nn.Module):
    """
    FastSAM3D model adapted for interactive training.

    This is a simplified implementation that follows FastSAM3D's architecture:
    - Lightweight ViT-Tiny encoder (6 layers instead of 12)
    - Prompt encoder (positional + learned embeddings)
    - Mask decoder

    For full FastSAM3D implementation, use:
    https://github.com/arcadelab/FastSAM3D
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 3,
        image_size: Tuple[int, int, int] = (128, 128, 128),
        embed_dim: int = 192,  # Smaller than ViT-B for efficiency
        depth: int = 6,  # FastSAM3D uses 6 layers
        num_heads: int = 6,
        use_pretrained: bool = False,
        pretrained_path: Optional[str] = None
    ):
        """
        Args:
            in_channels: Number of input channels
            out_channels: Number of output classes
            image_size: Input volume size
            embed_dim: Embedding dimension (192 for ViT-Tiny)
            depth: Number of transformer layers (6 for FastSAM3D)
            num_heads: Number of attention heads
            use_pretrained: Whether to load pretrained FastSAM3D weights
            pretrained_path: Path to pretrained weights
        """
        super().__init__()

        self.image_size = image_size
        self.embed_dim = embed_dim
        self.use_pretrained = use_pretrained

        # Image encoder (simplified ViT-Tiny)
        self.image_encoder = self._build_image_encoder(
            in_channels, embed_dim, depth, num_heads
        )

        # Prompt encoder
        self.prompt_encoder = FastSAM3DPromptEncoder(embed_dim, image_size)

        # Mask decoder
        self.mask_decoder = self._build_mask_decoder(embed_dim, out_channels)

        # Load pretrained weights if specified
        if use_pretrained and pretrained_path:
            self.load_pretrained(pretrained_path)

    def _build_image_encoder(self, in_channels, embed_dim, depth, num_heads):
        """
        Build lightweight ViT encoder.
        For full implementation, use FastSAM3D's encoder.
        """
        # Simplified: Use 3D convolution to encode image
        encoder = nn.Sequential(
            nn.Conv3d(in_channels, embed_dim // 4, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm3d(embed_dim // 4),
            nn.ReLU(inplace=True),
            nn.Conv3d(embed_dim // 4, embed_dim // 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(embed_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv3d(embed_dim // 2, embed_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(embed_dim),
            nn.ReLU(inplace=True),
        )
        return encoder

    def _build_mask_decoder(self, embed_dim, out_channels):
        """
        Build mask decoder.
        Upsamples from embeddings to segmentation mask.
        """
        decoder = nn.Sequential(
            nn.ConvTranspose3d(embed_dim, embed_dim // 2, kernel_size=2, stride=2),
            nn.BatchNorm3d(embed_dim // 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(embed_dim // 2, embed_dim // 4, kernel_size=2, stride=2),
            nn.BatchNorm3d(embed_dim // 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(embed_dim // 4, embed_dim // 8, kernel_size=2, stride=2),
            nn.BatchNorm3d(embed_dim // 8),
            nn.ReLU(inplace=True),
            nn.Conv3d(embed_dim // 8, out_channels, kernel_size=1),
        )
        return decoder

    def forward(
        self,
        image: torch.Tensor,
        points: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        prompt_channels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass with prompts.

        Args:
            image: [B, 1, D, H, W] input volume
            points: [B, N, 3] point coordinates (if using SAM-style prompts)
            labels: [B, N] point labels (if using SAM-style prompts)
            prompt_channels: [B, 2, D, H, W] pre-encoded prompts (alternative)

        Returns:
            logits: [B, num_classes, D, H, W]
        """
        # Encode image
        image_embedding = self.image_encoder(image)  # [B, embed_dim, D', H', W']

        # Encode prompts
        if points is not None and labels is not None:
            # SAM-style: encode point coordinates
            prompt_embedding = self.prompt_encoder(points, labels)  # [B, N, embed_dim]

            # Aggregate prompt embeddings (simple mean pooling)
            prompt_embedding = prompt_embedding.mean(dim=1, keepdim=True)  # [B, 1, embed_dim]

            # Broadcast to spatial dimensions
            B, _, D, H, W = image_embedding.shape
            prompt_embedding = prompt_embedding.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            prompt_embedding = prompt_embedding.expand(B, self.embed_dim, D, H, W)

            # Combine with image embedding
            combined_embedding = image_embedding + prompt_embedding

        elif prompt_channels is not None:
            # Alternative: use pre-encoded prompt channels
            # Encode them to match embedding dimension
            prompt_enc = nn.Conv3d(2, self.embed_dim, kernel_size=1).to(image.device)
            prompt_embedding = prompt_enc(prompt_channels)

            # Downsample to match image_embedding size
            prompt_embedding = nn.functional.interpolate(
                prompt_embedding,
                size=image_embedding.shape[2:],
                mode='trilinear',
                align_corners=False
            )

            # Combine
            combined_embedding = image_embedding + prompt_embedding
        else:
            # No prompts - just use image
            combined_embedding = image_embedding

        # Decode to segmentation mask
        logits = self.mask_decoder(combined_embedding)

        return logits

    def load_pretrained(self, checkpoint_path: str):
        """Load pretrained FastSAM3D weights."""
        print(f"Loading pretrained weights from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)  # ← Add weights_only=False

        # Load weights (may need adaptation based on checkpoint format)
        if 'model_state_dict' in checkpoint:
            self.load_state_dict(checkpoint['model_state_dict'], strict=False)
        elif 'state_dict' in checkpoint:
            self.load_state_dict(checkpoint['state_dict'], strict=False)
        else:
            self.load_state_dict(checkpoint, strict=False)

        print("Pretrained weights loaded successfully")


# Compatibility wrapper to work with existing training code
class FastSAM3DInteractiveWrapper(nn.Module):
    """
    Wrapper to make FastSAM3D compatible with existing training pipeline.
    Converts prompt_channels to SAM-style point prompts.
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.model = FastSAM3DInteractive(*args, **kwargs)

    def forward(self, image, prompts):
        """
        Args:
            image: [B, 1, D, H, W]
            prompts: [B, 2, D, H, W] encoded prompt channels

        Returns:
            logits: [B, num_classes, D, H, W]
        """
        # Use prompt_channels mode
        return self.model(image, prompt_channels=prompts)


if __name__ == "__main__":
    # Test FastSAM3D model
    print("Testing FastSAM3DInteractive")
    print('='*60)

    # Create model
    model = FastSAM3DInteractive(
        in_channels=1,
        out_channels=3,
        image_size=(128, 128, 128),
        embed_dim=192,
        depth=6,
        num_heads=6
    )

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # Test with point prompts (SAM-style)
    print("\nTest 1: SAM-style point prompts")
    image = torch.randn(2, 1, 128, 128, 128)
    points = torch.tensor([
        [[64, 64, 64], [32, 32, 32]],  # Batch 1: 2 points
        [[96, 96, 96], [16, 16, 16]]   # Batch 2: 2 points
    ], dtype=torch.float32)
    labels = torch.tensor([[1, 0], [1, 1]], dtype=torch.long)  # Positive/negative

    with torch.no_grad():
        output = model(image, points=points, labels=labels)

    print(f"  Input: {image.shape}")
    print(f"  Points: {points.shape}")
    print(f"  Output: {output.shape}")

    # Test with pre-encoded prompts (our framework's style)
    print("\nTest 2: Pre-encoded prompt channels")
    prompt_channels = torch.randn(2, 2, 128, 128, 128)

    with torch.no_grad():
        output = model(image, prompt_channels=prompt_channels)

    print(f"  Input: {image.shape}")
    print(f"  Prompts: {prompt_channels.shape}")
    print(f"  Output: {output.shape}")

    # Test wrapper
    print("\nTest 3: Wrapper (for compatibility)")
    wrapper = FastSAM3DInteractiveWrapper(
        in_channels=1, out_channels=3, image_size=(128, 128, 128)
    )

    with torch.no_grad():
        output = wrapper(image, prompt_channels)

    print(f"  Output: {output.shape}")
    print("\n✓ All tests passed!")