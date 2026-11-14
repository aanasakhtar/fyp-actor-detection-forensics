"""
End-to-end pipeline evaluation and latency metrics.
"""

from typing import List, Dict, Any
import numpy as np

def compute_pipeline_summary(
    video_results: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Compute aggregate metrics across all processed videos in the pipeline.
    """
    total_videos = len(video_results)
    if total_videos == 0:
        return {}
        
    videos_with_faces = sum(1 for r in video_results if r.get("faces_detected", 0) > 0)
    videos_with_actor_match = sum(1 for r in video_results if r.get("actor_matched", False))
    videos_flagged_altered = sum(1 for r in video_results if r.get("is_altered", False))
    
    latencies = [r.get("total_time_sec", 0.0) for r in video_results]
    
    return {
        "total_videos_analyzed": total_videos,
        "face_detection_rate": round(videos_with_faces / total_videos, 4),
        "actor_matching_rate": round(videos_with_actor_match / total_videos, 4),
        "manipulation_flag_rate": round(videos_flagged_altered / total_videos, 4),
        "mean_latency_sec": round(float(np.mean(latencies)), 2) if latencies else 0.0,
        "p95_latency_sec": round(float(np.percentile(latencies, 95)), 2) if latencies else 0.0
    }
