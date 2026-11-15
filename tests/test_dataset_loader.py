"""
Unit tests for FaceForensics++ metadata parsing and leakage-free splitting.
"""

import pytest
from pipeline.data.metadata import parse_ffpp_filename, parse_pair_mapping
from pipeline.data.dataset_loader import FFppDatasetLoader

def test_parse_ffpp_filename_original():
    target_id, source_id = parse_ffpp_filename("042.mp4")
    assert target_id == "042"
    assert source_id is None

def test_parse_ffpp_filename_manipulated():
    target_id, source_id = parse_ffpp_filename("042_189.mp4")
    assert target_id == "042"
    assert source_id == "189"

def test_parse_pair_mapping():
    meta = parse_pair_mapping("/path/to/Deepfakes/c23/videos/003_045.mp4", category="Deepfakes", compression="c23")
    assert meta.video_id == "003_045"
    assert meta.target_id == "003"
    assert meta.source_id == "045"
    assert meta.is_fake is True
    assert meta.is_manipulated is True
    assert meta.donor_id == "045"
    assert meta.path == "/path/to/Deepfakes/c23/videos/003_045.mp4"
    assert meta.compression == "c23"

def test_leakage_free_splits_synthetic():
    loader = FFppDatasetLoader("/nonexistent/path")
    # Manually populate mock videos
    for tid in ["001", "002", "003", "004", "005", "006", "007", "008", "009", "010"]:
        # Add original
        loader.videos.append(parse_pair_mapping(f"{tid}.mp4", category="original", compression="c23"))
        # Add fake
        loader.videos.append(parse_pair_mapping(f"{tid}_999.mp4", category="FaceSwap", compression="c23"))
        
    train, val, test = loader.get_leakage_free_splits(0.60, 0.20, 0.20, seed=42)
    
    train_targets = set(v.target_id for v in train)
    val_targets = set(v.target_id for v in val)
    test_targets = set(v.target_id for v in test)
    
    # Assert zero target ID overlap between any splits
    assert len(train_targets.intersection(val_targets)) == 0
    assert len(train_targets.intersection(test_targets)) == 0
    assert len(val_targets.intersection(test_targets)) == 0
    assert len(train_targets) + len(val_targets) + len(test_targets) == 10
