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
from pipeline.visualization.evaluation_plots import plot_similarity_distributions
from pipeline.utils.logging import get_logger

logger = get_logger("run_arcface")

def main():
    parser = argparse.ArgumentParser(description="Run ArcFace actor verification evaluation.")
    parser.add_argument("--data-dir", type=str, default="./data", help="Directory of videos.")
    parser.add_argument("--output-dir", type=str, default="./results/arcface", help="Output directory for metrics and plots.")
    parser.add_argument("--scrfd-dir", type=str, default=None, help="Directory containing SCRFD detection JSONs.")
    parser.add_argument("--max-frames", type=int, default=10, help="Frames sampled per video.")
    parser.add_argument("--face-selection", type=str, default="actor_match",
                        choices=["actor_match", "largest", "first"],
                        help="Face selection strategy: 'actor_match' (Identity Gating across all detected faces, default), 'largest' (spatial area), or 'first' (naive).")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    loader = FFppDatasetLoader(args.data_dir)
    detector = SCRFDDetector(conf_threshold=0.50)
    encoder = ArcFaceEncoder()
    gallery = ActorReferenceGallery()

    originals = loader.filter(categories=["original", "Celeb-real", "YouTube-real"])
    if not originals:
        # Fallback to any videos if specific categories not named original
        originals = [v for v in loader.videos if not v.is_fake] or loader.videos[:5]

    num_enroll = min(10, len(originals))
    logger.info(f"Enrolling {num_enroll} actors into reference gallery from pristine source videos...")
    
    for meta in originals[:num_enroll]:
        reader = VideoReader(meta.video_path)
        frames = reader.sample_uniform(max_frames=args.max_frames)
        embs = []
        for f in frames:
            dets = detector.detect_frame(f.image_bgr, f.frame_index, f.timestamp_sec)
            if not dets:
                continue
            # When enrolling reference actor, select the primary foreground face (largest area)
            best_det = max(dets, key=lambda d: (d.bbox[2] - d.bbox[0]) * (d.bbox[3] - d.bbox[1]))
            if best_det.landmarks_5 is not None:
                aligned = align_face_5point(f.image_bgr, best_det.landmarks_5)
                if aligned is not None:
                    emb = encoder.extract_embedding(aligned)
                    if emb is not None:
                        embs.append(emb)
        if embs:
            gallery.enroll_actor(meta.target_id, f"Actor_{meta.target_id}", embs[:5])

    enrolled_ids = list(gallery.prototypes.keys())
    logger.info(f"Successfully enrolled {len(enrolled_ids)} actors: {enrolled_ids}")

    genuine_scores = []
    impostor_scores = []
    
    logger.info(f"Evaluating verification protocol (face-selection='{args.face_selection}') across {len(loader)} videos...")

    for meta in loader.videos:
        reader = VideoReader(meta.video_path)
        frames = reader.sample_uniform(max_frames=max(4, args.max_frames // 2))
        for f in frames:
            dets = detector.detect_frame(f.image_bgr, f.frame_index, f.timestamp_sec)
            if not dets:
                continue

            target_proto = gallery.get_prototype(meta.target_id)
            chosen_emb = None

            if args.face_selection == "actor_match" and target_proto is not None:
                # Identity Gating: evaluate all detected faces (foreground & background) against registered prototype
                best_sim = -1.0
                best_emb = None
                for d in dets:
                    if d.landmarks_5 is not None:
                        aligned = align_face_5point(f.image_bgr, d.landmarks_5)
                        if aligned is not None:
                            emb = encoder.extract_embedding(aligned)
                            if emb is not None:
                                s = float(np.dot(emb, target_proto))
                                if s > best_sim:
                                    best_sim = s
                                    best_emb = emb
                if best_sim >= 0.40:
                    chosen_emb = best_emb
                else:
                    # Target actor not present in this frame (cutaway shot / non-target scene); isolate and skip
                    continue
            elif args.face_selection == "largest":
                # Spatial Saliency: select largest bounding box face (foreground actor)
                best_det = max(dets, key=lambda d: (d.bbox[2] - d.bbox[0]) * (d.bbox[3] - d.bbox[1]))
                if best_det.landmarks_5 is not None:
                    aligned = align_face_5point(f.image_bgr, best_det.landmarks_5)
                    if aligned is not None:
                        chosen_emb = encoder.extract_embedding(aligned)
            else:
                # Naive first detection
                if dets and dets[0].landmarks_5 is not None:
                    aligned = align_face_5point(f.image_bgr, dets[0].landmarks_5)
                    if aligned is not None:
                        chosen_emb = encoder.extract_embedding(aligned)

            if chosen_emb is None:
                continue

            emb = chosen_emb

            # 1. Compare against target identity prototype
            if target_proto is not None:
                sim = float(np.dot(emb, target_proto))
                # Genuine: original or morphology-preserving reenactment
                if not meta.is_fake or meta.category in ["Face2Face", "NeuralTextures"]:
                    genuine_scores.append(sim)
                else:
                    # Face swapped: target body, but donor face -> Impostor
                    impostor_scores.append(sim)

            # 2. Compare against non-target enrolled actors (cross-identity impostor test)
            for other_id, other_proto in gallery.prototypes.items():
                if other_id != meta.target_id:
                    cross_sim = float(np.dot(emb, other_proto))
                    impostor_scores.append(cross_sim)

    metrics = compute_verification_metrics(genuine_scores, impostor_scores)
    
    print("\n" + "="*70)
    print("      STAGE 2: ARCFACE ACTOR VERIFICATION EVALUATION RESULTS")
    print("="*70)
    print(f"Reference Gallery Enrolled:  {len(enrolled_ids)} Actors")
    print(f"Total Genuine Comparisons:   {len(genuine_scores)}")
    print(f"Total Impostor Comparisons:  {len(impostor_scores)}")
    print("-" * 70)
    print(f"ROC-AUC:                     {metrics['roc_auc']:.4f}")
    print(f"Equal Error Rate (EER):      {metrics['eer'] * 100:.2f}%")
    print(f"Optimal Cosine Threshold:    {metrics['optimal_threshold']:.4f}")
    print(f"Genuine Cosine Sim (Mean):   {metrics.get('genuine_mean', 0.0):.4f} ± {metrics.get('genuine_std', 0.0):.4f}")
    print(f"Impostor Cosine Sim (Mean):  {metrics.get('impostor_mean', 0.0):.4f} ± {metrics.get('impostor_std', 0.0):.4f}")
    print("="*70 + "\n")

    # Save metrics JSON
    metrics_path = os.path.join(args.output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[+] Saved verification metrics to: {metrics_path}")

    # Plot and save similarity distribution
    try:
        plot_path = os.path.join(args.output_dir, "arcface_similarity_distribution.png")
        plot_similarity_distributions(
            genuine_scores, 
            impostor_scores, 
            threshold=metrics["optimal_threshold"], 
            save_path=plot_path
        )
        print(f"[+] Saved similarity distribution plot to: {plot_path}")
    except Exception as e:
        logger.warning(f"Could not generate plot: {e}")

if __name__ == "__main__":
    main()
