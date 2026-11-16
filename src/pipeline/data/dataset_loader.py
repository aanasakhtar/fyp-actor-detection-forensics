"""
Dataset discovery, indexing, and zero-leakage partition splits for FaceForensics++.
"""

import os
import glob
from typing import List, Dict, Optional, Tuple
import numpy as np
from .metadata import FFppVideoMetadata, parse_pair_mapping
from ..utils.logging import get_logger

logger = get_logger("dataset_loader")

class FFppDatasetLoader:
    """Discovers and manages FaceForensics++ videos across categories and compressions."""
    
    def __init__(self, data_root: str):
        self.data_root = data_root
        self.videos: List[FFppVideoMetadata] = []
        self._scan_dataset()
        
    def _scan_dataset(self) -> None:
        """Scan the dataset directory for standard FF++ structure."""
        if not os.path.exists(self.data_root):
            logger.warning(f"Data root directory does not exist: {self.data_root}")
            return
            
        # Scan pattern: <data_root>/**/<category>/<compression>/videos/*.mp4
        mp4_files = glob.glob(os.path.join(self.data_root, "**", "*.mp4"), recursive=True)
        for filepath in mp4_files:
            # Determine category and compression from path components
            norm_path = os.path.normpath(filepath)
            parts = norm_path.split(os.sep)
            
            compression = "c23"
            for c in ["c23", "c40", "raw"]:
                if c in parts:
                    compression = c
                    break
                    
            category = "original"
            for cat in [
                "Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures", "FaceShifter", 
                "DeepFakeDetection", "Celeb-real", "Celeb-synthesis", "YouTube-real"
            ]:
                if cat in parts:
                    category = cat
                    break
            if ("youtube" in parts or "original_sequences" in parts) and "Celeb-real" not in parts and "YouTube-real" not in parts:
                category = "original"
                
            meta = parse_pair_mapping(filepath, category=category, compression=compression)
            self.videos.append(meta)
            
        logger.info(f"Scanned {len(self.videos)} total FF++ videos in {self.data_root}")

    def __len__(self) -> int:
        return len(self.videos)

    def filter(self, 
               categories: Optional[List[str]] = None, 
               compressions: Optional[List[str]] = None) -> List[FFppVideoMetadata]:
        """Filter videos by manipulation categories and compression levels."""
        filtered = self.videos
        if categories:
            cat_set = set(c.lower() for c in categories)
            filtered = [v for v in filtered if v.category.lower() in cat_set]
        if compressions:
            comp_set = set(c.lower() for c in compressions)
            filtered = [v for v in filtered if v.compression.lower() in comp_set]
        return filtered

    def get_leakage_free_splits(self, 
                                train_ratio: float = 0.70, 
                                val_ratio: float = 0.15, 
                                test_ratio: float = 0.15, 
                                seed: int = 42) -> Tuple[List[FFppVideoMetadata], List[FFppVideoMetadata], List[FFppVideoMetadata]]:
        """
        Partition videos into train, validation, and test splits strictly grouped by target_id.
        Guarantees that no original video or any of its manipulated counterparts appear across splits.
        """
        all_target_ids = sorted(list(set(v.target_id for v in self.videos)))
        rng = np.random.RandomState(seed)
        rng.shuffle(all_target_ids)
        
        n_total = len(all_target_ids)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        train_ids = set(all_target_ids[:n_train])
        val_ids = set(all_target_ids[n_train:n_train + n_val])
        test_ids = set(all_target_ids[n_train + n_val:])
        
        train_videos = [v for v in self.videos if v.target_id in train_ids]
        val_videos = [v for v in self.videos if v.target_id in val_ids]
        test_videos = [v for v in self.videos if v.target_id in test_ids]
        
        logger.info(f"Splits created: Train={len(train_videos)} videos ({len(train_ids)} IDs), "
                    f"Val={len(val_videos)} videos ({len(val_ids)} IDs), "
                    f"Test={len(test_videos)} videos ({len(test_ids)} IDs)")
        return train_videos, val_videos, test_videos
