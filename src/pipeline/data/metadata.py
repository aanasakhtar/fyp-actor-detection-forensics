"""
FaceForensics++ dataset filename and metadata parsing utilities.
"""

import os
from dataclasses import dataclass
from typing import Optional, Tuple

@dataclass
class FFppVideoMetadata:
    """Metadata container for a FaceForensics++ video."""
    video_path: str
    video_id: str
    category: str              # 'original', 'Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures', 'FaceShifter'
    compression: str           # 'c23', 'c40', 'raw'
    is_fake: bool              # False for original, True for manipulated
    target_id: str             # Base/target video identifier (e.g. '003')
    source_id: Optional[str]   # Donor face identifier (e.g. '045') or None if original

    @property
    def is_manipulated(self) -> bool:
        return self.is_fake

    @property
    def donor_id(self) -> Optional[str]:
        return self.source_id

    @property
    def path(self) -> str:
        return self.video_path

def parse_ffpp_filename(filename: str) -> Tuple[str, Optional[str]]:
    """
    Parse an FF++ video filename (e.g. '003.mp4' or '003_045.mp4').
    Returns (target_id, source_id).
    """
    base = os.path.splitext(os.path.basename(filename))[0]
    parts = base.split("_")
    if len(parts) == 1:
        return parts[0], None
    elif len(parts) >= 2:
        return parts[0], parts[1]
    return base, None

def parse_pair_mapping(video_path: str, category: str, compression: str) -> FFppVideoMetadata:
    """
    Construct full FFppVideoMetadata from video file path, category, and compression.
    """
    filename = os.path.basename(video_path)
    video_id = os.path.splitext(filename)[0]
    target_id, source_id = parse_ffpp_filename(filename)
    is_fake = category.lower() != "original" and "original" not in category.lower()
    
    return FFppVideoMetadata(
        video_path=video_path,
        video_id=video_id,
        category=category,
        compression=compression,
        is_fake=is_fake,
        target_id=target_id,
        source_id=source_id
    )
