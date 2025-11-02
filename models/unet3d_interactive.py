"""
3D U-Net Interactive Model

Standard 3D U-Net architecture modified to accept prompt channels as input.
"""

import torch
import torch.nn as nn
from typing import Tuple


class Conv3DBlock(nn.Module):
    """3D Convolution block with BatchNorm and ReLU."""

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size, padding=padding),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class UNet3DInteractive(nn.Module):
    """
    3D U-Net for interactive segmentation.

    Takes image + prompt channels as input:
        - Image: 1 channel (CT/MRI)
        - Prompts: 2 channels (positive points, negative points)

    Total input: 3 channels
    """

    def __init__(
        self,
        in_channels: int = 1,
        prompt_channels: int = 2,
        out_channels: int = 3,
        base_filters: int = 32,
        depth: int = 4
    ):
        """
        Args:
            in_channels: Number of image channels (1 for CT/MRI)
            prompt_channels: Number of prompt channels (2 for pos/neg)
            out_channels: Number of output classes
            base_filters: Number of filters in first layer
            depth: Depth of U-Net (number of down/up sampling stages)
        """
        super().__init__()

        self.in_channels = in_channels
        self.prompt_channels = prompt_channels
        self.out_channels = out_channels
        self.depth = depth

        # Total input channels = image + prompts
        total_in_channels = in_channels + prompt_channels

        # Encoder (downsampling path)
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()

        current_channels = total_in_channels
        for i in range(depth):
            out_ch = base_filters * (2 ** i)
            self.encoders.append(Conv3DBlock(current_channels, out_ch))
            self.pools.append(nn.MaxPool3d(2))
            current_channels = out_ch

        # Bottleneck
        self.bottleneck = Conv3DBlock(current_channels, base_filters * (2 ** depth))

        # Decoder (upsampling path)
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()

        for i in range(depth - 1, -1, -1):
            in_ch = base_filters * (2 ** (i + 1))
            out_ch = base_filters * (2 ** i)

            self.upconvs.append(
                nn.ConvTranspose3d(in_ch, out_ch, kernel_size=2, stride=2)
            )
            # *2 because of skip connection
            self.decoders.append(Conv3DBlock(out_ch * 2, out_ch))

        # Final output layer
        self.output = nn.Conv3d(base_filters, out_channels, kernel_size=1)

    def forward(self, image, prompts):
        """
        Forward pass.

        Args:
            image: [B, 1, D, H, W] CT/MRI volume
            prompts: [B, 2, D, H, W] encoded prompt channels

        Returns:
            logits: [B, num_classes, D, H, W] segmentation logits
        """
        # Concatenate image and prompts
        x = torch.cat([image, prompts], dim=1)  # [B, 3, D, H, W]

        # Encoder path with skip connections
        skip_connections = []
        for encoder, pool in zip(self.encoders, self.pools):
            x = encoder(x)
            skip_connections.append(x)
            x = pool(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder path with skip connections
        skip_connections = skip_connections[::-1]  # Reverse for decoder
        for i, (upconv, decoder, skip) in enumerate(
            zip(self.upconvs, self.decoders, skip_connections)
        ):
            x = upconv(x)

            # Handle potential size mismatch due to odd dimensions
            if x.shape != skip.shape:
                x = self._pad_to_match(x, skip)

            x = torch.cat([x, skip], dim=1)
            x = decoder(x)

        # Output
        logits = self.output(x)

        return logits

    def _pad_to_match(self, x, target):
        """Pad x to match target size."""
        diff_d = target.size(2) - x.size(2)
        diff_h = target.size(3) - x.size(3)
        diff_w = target.size(4) - x.size(4)

        padding = [
            diff_w // 2, diff_w - diff_w // 2,
            diff_h // 2, diff_h - diff_h // 2,
            diff_d // 2, diff_d - diff_d // 2,
        ]

        return nn.functional.pad(x, padding)


class UNet3DInteractiveLightning(nn.Module):
    """PyTorch Lightning wrapper for UNet3DInteractive (placeholder for now)."""

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.model = UNet3DInteractive(*args, **kwargs)

    def forward(self, image, prompts):
        return self.model(image, prompts)


if __name__ == "__main__":
    # Test the model
    print("Testing UNet3DInteractive")
    print('='*60)

    # Create model
    model = UNet3DInteractive(
        in_channels=1,
        prompt_channels=2,
        out_channels=3,
        base_filters=16,  # Smaller for testing
        depth=3
    )

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    # Test forward pass
    batch_size = 2
    image = torch.randn(batch_size, 1, 32, 64, 64)
    prompts = torch.randn(batch_size, 2, 32, 64, 64)

    print(f"\nInput shapes:")
    print(f"  Image: {image.shape}")
    print(f"  Prompts: {prompts.shape}")

    with torch.no_grad():
        output = model(image, prompts)

    print(f"\nOutput shape: {output.shape}")
    print(f"Expected: [{batch_size}, 3, 32, 64, 64]")

    # Test with different input sizes
    print(f"\n Testing with different patch sizes...")
    test_sizes = [(16, 32, 32), (32, 64, 64), (48, 96, 96)]

    for d, h, w in test_sizes:
        image = torch.randn(1, 1, d, h, w)
        prompts = torch.randn(1, 2, d, h, w)

        with torch.no_grad():
            output = model(image, prompts)

        print(f"  Input: [1, 1, {d}, {h}, {w}] -> Output: {list(output.shape)}")