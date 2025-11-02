"""
Base Prompt Encoder Interface
All prompt encoding methods should inherit from this class.
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import List, Tuple, Optional


class BasePromptEncoder(ABC):
    """
    Abstract base class for prompt encoders.

    All prompt encoders should implement the encode() method which takes
    positive and negative point coordinates and returns encoded prompt channels.
    """

    def __init__(self, name: str = "base"):
        """
        Args:
            name: Identifier for this encoding method
        """
        self.name = name

    @abstractmethod
    def encode(
        self,
        positive_points: List[Tuple[int, int, int]],
        negative_points: List[Tuple[int, int, int]],
        volume_shape: Tuple[int, int, int],
        spacing: Optional[Tuple[float, float, float]] = None
    ) -> np.ndarray:
        """
        Encode user clicks into prompt channels.

        Args:
            positive_points: List of (z, y, x) coordinates for positive clicks
            negative_points: List of (z, y, x) coordinates for negative clicks
            volume_shape: Shape of the volume (D, H, W)
            spacing: Physical spacing between voxels (optional)

        Returns:
            prompt_channels: Array of shape [2, D, H, W] where
                             prompt_channels[0] = positive channel
                             prompt_channels[1] = negative channel
        """
        pass

    def validate_points(
        self,
        points: List[Tuple[int, int, int]],
        volume_shape: Tuple[int, int, int]
    ) -> List[Tuple[int, int, int]]:
        """
        Validate that points are within volume bounds.

        Args:
            points: List of (z, y, x) coordinates
            volume_shape: Shape of the volume (D, H, W)

        Returns:
            valid_points: List of validated points within bounds
        """
        valid_points = []
        D, H, W = volume_shape

        for z, y, x in points:
            if 0 <= z < D and 0 <= y < H and 0 <= x < W:
                valid_points.append((z, y, x))
            else:
                print(f"Warning: Point ({z}, {y}, {x}) is out of bounds for volume {volume_shape}")

        return valid_points

    def __repr__(self):
        return f"{self.__class__.__name__}(name='{self.name}')"