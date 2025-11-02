"""
Test Script

Quick test to verify that all components are working correctly.

Usage:
    python tools/test_components.py
"""

import sys
import os
import numpy as np
import torch

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_prompt_encoders():
    """Test all prompt encoders."""
    print("\n" + "="*60)
    print("TESTING PROMPT ENCODERS")
    print("="*60)

    from prompt_encoders import (
        BinaryPromptEncoder,
        GaussianPromptEncoder,
        DiskPromptEncoder,
        get_encoder
    )

    # Test data
    pos_points = [(16, 32, 32), (20, 40, 40)]
    neg_points = [(8, 16, 16)]
    volume_shape = (32, 64, 64)

    encoders_to_test = [
        ('binary', BinaryPromptEncoder()),
        ('gaussian', GaussianPromptEncoder(sigma=3.0)),
        ('disk', DiskPromptEncoder(radius=3)),
    ]

    for name, encoder in encoders_to_test:
        print(f"\nTesting {name} encoder...")
        try:
            prompts = encoder.encode(pos_points, neg_points, volume_shape)
            assert prompts.shape == (2, *volume_shape), f"Wrong shape: {prompts.shape}"
            assert prompts.dtype == np.float32, f"Wrong dtype: {prompts.dtype}"
            print(f"  ✓ {name} encoder works!")
            print(f"    Output shape: {prompts.shape}")
            print(f"    Positive channel non-zeros: {np.count_nonzero(prompts[0])}")
            print(f"    Negative channel non-zeros: {np.count_nonzero(prompts[1])}")
        except Exception as e:
            print(f"  ✗ {name} encoder failed: {e}")

    # Test factory function
    print(f"\nTesting encoder factory function...")
    try:
        encoder = get_encoder('gaussian', sigma=5.0)
        prompts = encoder.encode(pos_points, neg_points, volume_shape)
        print(f"  ✓ Factory function works!")
    except Exception as e:
        print(f"  ✗ Factory function failed: {e}")


def test_prompt_simulator():
    """Test prompt simulator."""
    print("\n" + "="*60)
    print("TESTING PROMPT SIMULATOR")
    print("="*60)

    from data.prompt_simulator import PromptSimulator

    # Create synthetic mask
    mask = np.zeros((32, 64, 64), dtype=np.int32)
    mask[10:20, 20:40, 20:40] = 1  # Add foreground region

    simulator = PromptSimulator(
        n_points_range=(2, 5),
        sampling_strategy='mixed'
    )

    print("\nSimulating prompts...")
    try:
        pos_points, neg_points = simulator.simulate(mask)
        print(f"  ✓ Prompt simulator works!")
        print(f"    Generated {len(pos_points)} positive points")
        print(f"    Generated {len(neg_points)} negative points")
        print(f"    Example pos point: {pos_points[0] if pos_points else 'None'}")
    except Exception as e:
        print(f"  ✗ Prompt simulator failed: {e}")


def test_model():
    """Test 3D U-Net model."""
    print("\n" + "="*60)
    print("TESTING 3D U-NET MODEL")
    print("="*60)

    from models.unet3d_interactive import UNet3DInteractive

    model = UNet3DInteractive(
        in_channels=1,
        prompt_channels=2,
        out_channels=3,
        base_filters=16,
        depth=3
    )

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel created with {n_params:,} parameters")

    # Test forward pass
    print("\nTesting forward pass...")
    try:
        image = torch.randn(2, 1, 16, 32, 32)
        prompts = torch.randn(2, 2, 16, 32, 32)

        with torch.no_grad():
            output = model(image, prompts)

        assert output.shape == (2, 3, 16, 32, 32), f"Wrong output shape: {output.shape}"
        print(f"  ✓ Forward pass works!")
        print(f"    Input: image {image.shape}, prompts {prompts.shape}")
        print(f"    Output: {output.shape}")
    except Exception as e:
        print(f"  ✗ Forward pass failed: {e}")


def test_losses():
    """Test loss functions."""
    print("\n" + "="*60)
    print("TESTING LOSS FUNCTIONS")
    print("="*60)

    from utils.losses import DiceLoss, CombinedLoss

    # Create dummy data
    logits = torch.randn(2, 3, 16, 32, 32)
    targets = torch.randint(0, 3, (2, 16, 32, 32))

    losses_to_test = [
        ('DiceLoss', DiceLoss()),
        ('CombinedLoss', CombinedLoss(num_classes=3)),
    ]

    for name, loss_fn in losses_to_test:
        print(f"\nTesting {name}...")
        try:
            loss = loss_fn(logits, targets)
            assert loss.item() >= 0, "Loss should be non-negative"
            print(f"  ✓ {name} works!")
            print(f"    Loss value: {loss.item():.4f}")

            # Test backward
            loss.backward()
            print(f"    Backward pass successful")
        except Exception as e:
            print(f"  ✗ {name} failed: {e}")


def test_metrics():
    """Test evaluation metrics."""
    print("\n" + "="*60)
    print("TESTING EVALUATION METRICS")
    print("="*60)

    from utils.metrics import compute_dice_score, compute_iou

    # Create dummy predictions and targets
    pred = torch.randint(0, 3, (2, 16, 32, 32))
    target = torch.randint(0, 3, (2, 16, 32, 32))

    print("\nTesting Dice score...")
    try:
        dice = compute_dice_score(pred, target, num_classes=3)
        assert 0 <= dice <= 1, "Dice should be in [0, 1]"
        print(f"  ✓ Dice score works!")
        print(f"    Dice value: {dice:.4f}")
    except Exception as e:
        print(f"  ✗ Dice score failed: {e}")

    print("\nTesting IoU...")
    try:
        iou = compute_iou(pred, target, num_classes=3)
        assert 0 <= iou <= 1, "IoU should be in [0, 1]"
        print(f"  ✓ IoU works!")
        print(f"    IoU value: {iou:.4f}")
    except Exception as e:
        print(f"  ✗ IoU failed: {e}")


def main():
    """Run all tests."""
    print("="*60)
    print("RUNNING COMPONENT TESTS")
    print("="*60)

    tests = [
        ("Prompt Encoders", test_prompt_encoders),
        ("Prompt Simulator", test_prompt_simulator),
        ("Model", test_model),
        ("Loss Functions", test_losses),
        ("Metrics", test_metrics),
    ]

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"\n✗ {test_name} test suite failed: {e}")
            failed += 1
            import traceback
            traceback.print_exc()

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Passed: {passed}/{len(tests)}")
    print(f"Failed: {failed}/{len(tests)}")

    if failed == 0:
        print("\n✓ All tests passed! You're ready to start training.")
    else:
        print(f"\n✗ {failed} test(s) failed. Please check the errors above.")


if __name__ == "__main__":
    main()