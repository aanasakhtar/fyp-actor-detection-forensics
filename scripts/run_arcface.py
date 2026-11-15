"""
Standalone script to evaluate ArcFace actor verification proxy protocol.
"""

import os
import sys

# Ensure local 'src' directory takes priority in sys.path
src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

import argparse
import json
import numpy as np
from pipeline.data.dataset_loader import FFppDatasetLoader
from pipeline.data.video_reader import VideoReader
from pipeline.detection.scrfd_detector import SCRFDDetector
from pipeline.preprocessing.face_alignment import align_face_5point
from pipeline.recognition.arcface_encoder import ArcFaceEncoder
from pipeline.recognition.reference_gallery import ActorReferenceGallery
from pipeline.recognition.identity_matching import IdentityMatcher
from pipeline.evaluation.recognition_metrics import compute_verification_metrics
from pipeline.utils.logging import get_logger

logger = get_logger("run_arcface")

def main():
    parser = argparse.ArgumentParser(description="Run ArcFace actor verification evaluation.")
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--output-dir", type=str, default="./results/arcface")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    loader = FFppDatasetLoader(args.data_dir)
    detector = SCRFDDetector()
    encoder = ArcFaceEncoder()
    gallery = ActorReferenceGallery()

    originals = loader.filter(categories=["original"])
    logger.info(f"Enrolling {min(10, len(originals))} actors into reference gallery...")
    
    actor_embeddings = {}
    for meta in originals[:10]:
        reader = VideoReader(meta.video_path)
        frames = reader.sample_uniform(max_frames=10)
        embs = []
        for f in frames:
            dets = detector.detect_frame(f.image_bgr, f.frame_index, f.timestamp_sec)
            if dets and dets[0].landmarks_5 is not None:
                aligned = align_face_5point(f.image_bgr, dets[0].landmarks_5)
                if aligned is not None:
                    emb = encoder.extract_embedding(aligned)
                    if emb is not None:
                        embs.append(emb)
        if embs:
            actor_embeddings[meta.target_id] = embs
            gallery.enroll_actor(meta.target_id, f"Actor_{meta.target_id}", embs[:5])

    genuine_scores = []
    impostor_scores = []
    
    for meta in loader.videos:
        reader = VideoReader(meta.video_path)
        frames = reader.sample_uniform(max_frames=5)
        for f in frames:
            dets = detector.detect_frame(f.image_bgr)
            if dets and dets[0].landmarks_5 is not None:
                aligned = align_face_5point(f.image_bgr, dets[0].landmarks_5)
                if aligned is not None:
                    emb = encoder.extract_embedding(aligned)
                    if emb is not None:
                        proto = gallery.get_prototype(meta.target_id)
                        if proto is not None:
                            sim = float(np.dot(emb, proto))
                            if not meta.is_fake or meta.category in ["Face2Face", "NeuralTextures"]:
                                genuine_scores.append(sim)
                            else:
                                impostor_scores.append(sim)

    metrics = compute_verification_metrics(genuine_scores, impostor_scores)
    logger.info(f"ArcFace Verification Metrics: {metrics}")
    
    with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

if __name__ == "__main__":
    main()
