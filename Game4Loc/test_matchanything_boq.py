#!/usr/bin/env python3
"""
Test script for MatchAnything + BoQ model integration.

This script tests:
1. Model instantiation
2. Checkpoint loading
3. Forward pass with correct output shapes
4. Trainable parameter count
"""

import torch
import sys
sys.path.insert(0, '/media/aniel/storage/dataset_to_analize/GTA-UAV/Game4Loc')

from game4loc.models.model_matchanything_boq import DesModelMatchAnythingBoQ


def test_model_instantiation():
    """Test that the model can be instantiated."""
    print("=" * 80)
    print("TEST 1: Model Instantiation")
    print("=" * 80)

    try:
        model = DesModelMatchAnythingBoQ(
            checkpoint_path='/media/aniel/storage/dataset_to_analize/MatchAnything/matchanything_eloftr.ckpt',
            img_size=384,
            share_weights=True,
            num_queries=64,
            num_layers=2,
            mlp_output_dim=512
        )
        print("✓ Model instantiated successfully")
        return model
    except Exception as e:
        print(f"✗ Model instantiation failed: {e}")
        raise


def test_parameter_freezing(model):
    """Test that backbone parameters are frozen and BoQ parameters are trainable."""
    print("\n" + "=" * 80)
    print("TEST 2: Parameter Freezing")
    print("=" * 80)

    backbone_params = 0
    backbone_trainable = 0
    boq_params = 0
    boq_trainable = 0
    logit_scale_params = 1  # logit_scale is a single parameter

    for name, param in model.named_parameters():
        if 'backbone' in name:
            backbone_params += param.numel()
            if param.requires_grad:
                backbone_trainable += param.numel()
        elif 'aggregator' in name:
            boq_params += param.numel()
            if param.requires_grad:
                boq_trainable += param.numel()

    print(f"Backbone parameters: {backbone_params:,}")
    print(f"  - Trainable: {backbone_trainable:,}")
    print(f"  - Frozen: {backbone_params - backbone_trainable:,}")

    print(f"\nBoQ parameters: {boq_params:,}")
    print(f"  - Trainable: {boq_trainable:,}")

    print(f"\nLogit scale parameters: {logit_scale_params:,} (trainable)")

    total_params = backbone_params + boq_params + logit_scale_params
    total_trainable = backbone_trainable + boq_trainable + logit_scale_params

    print(f"\nTotal parameters: {total_params:,}")
    print(f"Total trainable: {total_trainable:,} ({100 * total_trainable / total_params:.2f}%)")

    # Assertions
    assert backbone_trainable == 0, "Backbone should be fully frozen"
    assert boq_trainable == boq_params, "BoQ should be fully trainable"
    print("\n✓ Parameter freezing correct")


def test_forward_pass_single(model):
    """Test forward pass with a single image."""
    print("\n" + "=" * 80)
    print("TEST 3: Forward Pass (Single Image)")
    print("=" * 80)

    batch_size = 4
    img_size = 384

    # Create dummy RGB image
    img = torch.randn(batch_size, 3, img_size, img_size)
    print(f"Input shape: {img.shape}")

    try:
        with torch.no_grad():
            output = model(img1=img)

        print(f"Output shape: {output.shape}")

        # Check output shape
        expected_shape = (batch_size, 512)
        assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"

        # Check L2 normalization (norm should be ~1.0)
        norms = torch.norm(output, p=2, dim=1)
        print(f"L2 norms: min={norms.min():.4f}, max={norms.max():.4f}, mean={norms.mean():.4f}")
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), "Output should be L2 normalized"

        print("✓ Single image forward pass successful")
        return output
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        raise


def test_forward_pass_dual(model):
    """Test forward pass with dual images (query + gallery)."""
    print("\n" + "=" * 80)
    print("TEST 4: Forward Pass (Dual Images)")
    print("=" * 80)

    batch_size = 4
    img_size = 384

    # Create dummy RGB images
    img1 = torch.randn(batch_size, 3, img_size, img_size)
    img2 = torch.randn(batch_size, 3, img_size, img_size)
    print(f"Input 1 shape: {img1.shape}")
    print(f"Input 2 shape: {img2.shape}")

    try:
        with torch.no_grad():
            emb1, emb2 = model(img1=img1, img2=img2)

        print(f"Output 1 shape: {emb1.shape}")
        print(f"Output 2 shape: {emb2.shape}")

        # Check output shapes
        expected_shape = (batch_size, 512)
        assert emb1.shape == expected_shape, f"Expected {expected_shape}, got {emb1.shape}"
        assert emb2.shape == expected_shape, f"Expected {expected_shape}, got {emb2.shape}"

        # Check L2 normalization
        norms1 = torch.norm(emb1, p=2, dim=1)
        norms2 = torch.norm(emb2, p=2, dim=1)
        print(f"Embedding 1 norms: min={norms1.min():.4f}, max={norms1.max():.4f}")
        print(f"Embedding 2 norms: min={norms2.min():.4f}, max={norms2.max():.4f}")

        assert torch.allclose(norms1, torch.ones_like(norms1), atol=1e-5), "Emb1 should be L2 normalized"
        assert torch.allclose(norms2, torch.ones_like(norms2), atol=1e-5), "Emb2 should be L2 normalized"

        print("✓ Dual image forward pass successful")
        return emb1, emb2
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        raise


def test_grayscale_conversion(model):
    """Test RGB to grayscale conversion."""
    print("\n" + "=" * 80)
    print("TEST 5: RGB to Grayscale Conversion")
    print("=" * 80)

    # Create a test RGB image
    rgb = torch.tensor([[[[1.0, 0.0],
                          [0.0, 1.0]],
                         [[0.0, 1.0],
                          [1.0, 0.0]],
                         [[0.5, 0.5],
                          [0.5, 0.5]]]])  # [1, 3, 2, 2]

    print(f"RGB input shape: {rgb.shape}")

    try:
        gray = model.rgb_to_grayscale(rgb)
        print(f"Grayscale output shape: {gray.shape}")

        # Check shape
        expected_shape = (1, 1, 2, 2)
        assert gray.shape == expected_shape, f"Expected {expected_shape}, got {gray.shape}"

        # Manual calculation for pixel [0, 0]:
        # R=1.0, G=0.0, B=0.5 -> 0.299*1.0 + 0.587*0.0 + 0.114*0.5 = 0.356
        expected_value = 0.299 * 1.0 + 0.587 * 0.0 + 0.114 * 0.5
        actual_value = gray[0, 0, 0, 0].item()
        print(f"Test pixel value: expected={expected_value:.4f}, actual={actual_value:.4f}")

        assert abs(actual_value - expected_value) < 1e-5, "Grayscale conversion incorrect"

        print("✓ Grayscale conversion correct")
    except Exception as e:
        print(f"✗ Grayscale conversion failed: {e}")
        raise


def test_spatial_to_sequence(model):
    """Test spatial to sequence conversion."""
    print("\n" + "=" * 80)
    print("TEST 6: Spatial to Sequence Conversion")
    print("=" * 80)

    # Create dummy spatial features [B, C, H, W]
    batch_size = 2
    channels = 256
    height = 48
    width = 48

    feats_c = torch.randn(batch_size, channels, height, width)
    print(f"Spatial features shape: {feats_c.shape}")

    try:
        feats_seq = model.spatial_to_sequence(feats_c)
        print(f"Sequence features shape: {feats_seq.shape}")

        # Check shape
        expected_shape = (batch_size, height * width, channels)
        assert feats_seq.shape == expected_shape, f"Expected {expected_shape}, got {feats_seq.shape}"

        print(f"Sequence length: {height * width} patches")
        print("✓ Spatial to sequence conversion correct")
    except Exception as e:
        print(f"✗ Spatial to sequence conversion failed: {e}")
        raise


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("MatchAnything + BoQ Model Test Suite")
    print("=" * 80)

    # Test 1: Instantiation
    model = test_model_instantiation()

    # Test 2: Parameter freezing
    test_parameter_freezing(model)

    # Test 3: Single image forward pass
    test_forward_pass_single(model)

    # Test 4: Dual image forward pass
    test_forward_pass_dual(model)

    # Test 5: Grayscale conversion
    test_grayscale_conversion(model)

    # Test 6: Spatial to sequence
    test_spatial_to_sequence(model)

    print("\n" + "=" * 80)
    print("ALL TESTS PASSED ✓")
    print("=" * 80)
    print("\nModel is ready for training!")
    print("\nTo train the model, use:")
    print("python train_gta.py \\")
    print("  --with_matchanything_boq \\")
    print("  --data_root /path/to/dataset \\")
    print("  --train_pairs_meta_file train.json \\")
    print("  --test_pairs_meta_file val.json \\")
    print("  --num_queries 64 \\")
    print("  --mlp_output_dim 512 \\")
    print("  --img_size 384 \\")
    print("  --with_weight \\")
    print("  --k 5 \\")
    print("  --epochs 10 \\")
    print("  --lr 0.0001 \\")
    print("  --batch_size 20")


if __name__ == '__main__':
    main()
