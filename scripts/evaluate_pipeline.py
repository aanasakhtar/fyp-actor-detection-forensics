"""
Integrated end-to-end pipeline evaluation script.
"""

import os
import argparse
import json
import time
from pipeline.data.dataset_loader import FFppDatasetLoader
from pipeline.data.video_reader import VideoReader
from pipeline.detection.scrfd_detector import SCRFDDetector
from pipeline.tracking.face_tracking import IoUFaceTracker
from pipeline.preprocessing.face_alignment import align_face_5point
from pipeline.preprocessing.face_cropping import crop_context_face
from pipeline.recognition.arcface_encoder import ArcFaceEncoder
from pipeline.recognition.reference_gallery import ActorReferenceGallery
from pipeline.recognition.identity_matching import IdentityMatcher
from pipeline.forensics.clip_detector import GenDCLIPDetector
from pipeline.evaluation.pipeline_metrics import compute_pipeline_summary
from pipeline.utils.logging import get_logger

logger = get_logger("evaluate_pipeline")

def main():
    parser = argparse.ArgumentParser(description="Run end-to-end CV evaluation pipeline.")
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--output-dir", type=str, default="./results/pipeline")
    parser.add_argument("--max-frames", type=int, default=16)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    loader = FFppDatasetLoader(args.data_dir)
    
    # Initialize pipeline components
    scrfd = SCRFDDetector()
    tracker = IoUFaceTracker()
    arcface = ArcFaceEncoder()
    gallery = ActorReferenceGallery()
    matcher = IdentityMatcher(gallery)
    gend = GenDCLIPDetector()

    # Enroll original videos as registered actors
    originals = loader.filter(categories=["original"])
    for meta in originals[:5]:
        reader = VideoReader(meta.video_path)
        frames = reader.sample_uniform(max_frames=5)
        embs = []
        for f in frames:
            dets = scrfd.detect_frame(f.image_bgr)
            if dets and dets[0].landmarks_5 is not None:
                aligned = align_face_5point(f.image_bgr, dets[0].landmarks_5)
                if aligned is not None:
                    emb = arcface.extract_embedding(aligned)
                    if emb is not None:
                        embs.append(emb)
        if embs:
            gallery.enroll_actor(meta.target_id, f"Actor_{meta.target_id}", embs)

    results = []
    logger.info(f"Executing end-to-end pipeline on {len(loader)} videos...")

    for meta in loader.videos:
        start_t = time.time()
        reader = VideoReader(meta.video_path)
        frames = reader.sample_uniform(max_frames=args.max_frames)
        
        detected_faces = 0
        actor_matched = False
        gend_scores = []

        for frame in frames:
            dets = scrfd.detect_frame(frame.image_bgr, frame.frame_index, frame.timestamp_sec)
            dets = tracker.update(dets, frame.frame_index)
            detected_faces += len(dets)
            
            for d in dets:
                # ArcFace path
                if d.landmarks_5 is not None:
                    aligned = align_face_5point(frame.image_bgr, d.landmarks_5)
                    if aligned is not None:
                        emb = arcface.extract_embedding(aligned)
                        if emb is not None:
                            match_res = matcher.match(emb)
                            if match_res.status == "possible match":
                                actor_matched = True

                # GenD path
                c = crop_context_face(frame.image_bgr, d.bbox)
                if c is not None:
                    probs = gend.predict_crops([c])
                    if probs:
                        gend_scores.append(probs[0])

        total_time = time.time() - start_t
        video_alteration_score = gend.aggregate_scores(gend_scores)
        is_altered = video_alteration_score >= 0.50

        results.append({
            "video_id": meta.video_id,
            "category": meta.category,
            "compression": meta.compression,
            "ground_truth_fake": meta.is_fake,
            "faces_detected": detected_faces,
            "actor_matched": actor_matched,
            "manipulation_score": round(video_alteration_score, 4),
            "is_altered": is_altered,
            "total_time_sec": round(total_time, 2)
        })

    summary = compute_pipeline_summary(results)
    logger.info(f"Pipeline Execution Summary: {summary}")

    with open(os.path.join(args.output_dir, "pipeline_summary.json"), "w") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)

if __name__ == "__main__":
    main()
