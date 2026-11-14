"""
High-level video frame extraction wrapper.
"""

from typing import List
from ..data.video_reader import VideoReader, SampledFrame

def extract_frames_from_video(video_path: str, max_frames: int = 32) -> List[SampledFrame]:
    """Helper function to decode and uniformly sample frames from a video path."""
    reader = VideoReader(video_path)
    return reader.sample_uniform(max_frames=max_frames)
