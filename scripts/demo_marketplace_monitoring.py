"""
End-to-End Talent Marketplace AI Likeness Monitoring Demo.
Simulates the complete production workflow:
  1. Onboarding / Enrollment: Ingests an actor's official studio portrait from 'ashraq/tmdb-people-image'.
  2. Stage 1 (SCRFD): Localizes facial bounding box and 5-point alignment keypoints.
  3. Stage 2 (ArcFace): Biometrically verifies if the actor in an in-the-wild Celeb-DF video matches the enrolled talent.
  4. Stage 3 (GenD): Classifies the video into Authentic (Real) or Unauthorized AI Deepfake (Manipulated).
  5. Executive Diagnostic Report: Generates a 3-panel forensic audit card (marketplace_audit_report.png).
"""

import os
import sys
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image

# Ensure local 'src' directory takes priority in sys.path
src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from pipeline.detection.scrfd_detector import SCRFDDetector
from pipeline.preprocessing.face_alignment import align_face_5point, align_face_gend
from pipeline.preprocessing.face_cropping import crop_context_face
from pipeline.recognition.arcface_encoder import ArcFaceEncoder
from pipeline.forensics.clip_detector import GenDCLIPDetector
from pipeline.forensics.pe_detector import GenDPEDetector
from pipeline.forensics.dinov3_detector import GenDDINOv3Detector
from pipeline.forensics.sbi_detector import SBIEfficientNetDetector
from pipeline.forensics.vit_detector import ViTDeepfakeDetector
from pipeline.data.video_reader import VideoReader
from pipeline.utils.logging import get_logger

logger = get_logger("marketplace_monitoring")


def fetch_official_tmdb_talent(
    actor_name: str,
    api_key: str = "b2b7416d572f285d036cca47fb16a1c4",
    detector: SCRFDDetector = None,
    encoder: ArcFaceEncoder = None,
    query_emb: np.ndarray = None
):
    """
    Query the official TMDB API to enroll the talent's multi-headshot portfolio gallery.
    Downloads all official studio portraits, extracts ArcFace vectors, and identifies
    the best matching reference headshot against the monitored video prototype.
    """
    import urllib.request
    import urllib.parse
    import json
    import io

    encoded_name = urllib.parse.quote(actor_name)
    search_url = f"https://api.themoviedb.org/3/search/person?api_key={api_key}&query={encoded_name}"
    
    try:
        req = urllib.request.Request(search_url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            results = data.get("results", [])
            if not results:
                logger.warning(f"No person found in official TMDB API for '{actor_name}'")
                return None
            person = results[0]
            p_id = person["id"]
            matched_name = person["name"]
            popularity = person.get("popularity", 0.0)

        # Fetch IMDb external ID
        imdb_id = "N/A"
        try:
            ext_url = f"https://api.themoviedb.org/3/person/{p_id}/external_ids?api_key={api_key}"
            with urllib.request.urlopen(urllib.request.Request(ext_url, headers={"User-Agent": "Mozilla/5.0"})) as resp:
                ext_data = json.loads(resp.read().decode("utf-8"))
                imdb_id = ext_data.get("imdb_id", "N/A")
        except Exception:
            pass

        # Fetch all official profiles/headshots
        img_url = f"https://api.themoviedb.org/3/person/{p_id}/images?api_key={api_key}"
        with urllib.request.urlopen(urllib.request.Request(img_url, headers={"User-Agent": "Mozilla/5.0"})) as resp:
            img_data = json.loads(resp.read().decode("utf-8"))
            profiles = img_data.get("profiles", [])

        if not profiles:
            logger.warning(f"No profile headshots found on TMDB for {matched_name}")
            return None

        logger.info(f"Official TMDB API: Ingesting {len(profiles)} portfolio headshots for {matched_name} (IMDb: {imdb_id})...")

        gallery_items = []
        for i, p in enumerate(profiles):
            file_path = p.get("file_path")
            if not file_path:
                continue
            photo_url = f"https://image.tmdb.org/t/p/original{file_path}"
            try:
                with urllib.request.urlopen(urllib.request.Request(photo_url, headers={"User-Agent": "Mozilla/5.0"})) as resp:
                    raw_bytes = resp.read()
                    pil_img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
                    img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

                    emb = None
                    if detector is not None and encoder is not None:
                        dets = detector.detect_frame(img_bgr)
                        if dets:
                            best_det = max(dets, key=lambda d: (d.bbox[2] - d.bbox[0]) * (d.bbox[3] - d.bbox[1]))
                            if best_det.landmarks_5 is not None:
                                aligned = align_face_5point(img_bgr, best_det.landmarks_5)
                                if aligned is not None:
                                    emb = encoder.extract_embedding(aligned)

                    sim = float(np.dot(emb, query_emb)) if (emb is not None and query_emb is not None) else 1.0
                    gallery_items.append({
                        "image_pil": pil_img,
                        "embedding": emb,
                        "sim": sim,
                        "url": photo_url
                    })
                    if query_emb is not None and emb is not None:
                        logger.info(f"  * Headshot {i+1}/{len(profiles)} Biometric Likeness: {sim:.4f}")
            except Exception as e:
                logger.warning(f"Could not load TMDB headshot {i}: {e}")

        if not gallery_items:
            return None

        # Select portrait with highest similarity to monitored face
        best_item = max(gallery_items, key=lambda g: g["sim"])
        gallery_embs = [g["embedding"] for g in gallery_items if g["embedding"] is not None]
        
        centroid_emb = None
        if gallery_embs:
            centroid_emb = np.mean(gallery_embs, axis=0)
            centroid_emb = centroid_emb / np.linalg.norm(centroid_emb)

        return {
            "name": matched_name,
            "imdb_id": imdb_id,
            "popularity": popularity,
            "image_pil": best_item["image_pil"],
            "best_emb": best_item["embedding"],
            "centroid_emb": centroid_emb,
            "max_sim": best_item["sim"],
            "all_sims": [g["sim"] for g in gallery_items],
            "num_headshots": len(gallery_items),
            "source": f"Official TMDB Multi-Headshot Gallery ({len(gallery_items)} images)"
        }
    except Exception as e:
        logger.error(f"Error querying official TMDB API: {e}")
        return None


def fetch_tmdb_actor(actor_name: str = None, top_k: int = 80, query_emb: np.ndarray = None, encoder: ArcFaceEncoder = None, detector: SCRFDDetector = None):
    """
    Search or stream actors from 'ashraq/tmdb-people-image'.
    If actor_name is given, searches for exact name match.
    If query_emb is given, scans top popular actors and returns the best biometric match.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("Please install datasets via: pip install datasets")
        return None

    logger.info("Streaming 'ashraq/tmdb-people-image' from Hugging Face...")
    ds = load_dataset("ashraq/tmdb-people-image", split="train", streaming=True)

    candidates = []
    best_match = None
    best_sim = -1.0

    count = 0
    for entry in ds:
        name = entry.get("name", "")
        dept = entry.get("known_for_department", "")
        pop = float(entry.get("popularity", 0.0))
        img = entry.get("image")

        # Case A: Exact or partial name search
        if actor_name:
            if actor_name.lower() in name.lower():
                logger.info(f"Found requested actor in TMDB: {name} (Popularity: {pop:.1f}, IMDb: {entry.get('imdb_id')})")
                return {
                    "name": name,
                    "imdb_id": entry.get("imdb_id", "N/A"),
                    "popularity": pop,
                    "image_pil": img,
                    "cosine_sim": 1.0
                }
            continue

        # Quality filter for auto-identification across candidate stream
        if dept != "Acting" and pop < 10.0:
            continue

        # Case B: Biometric cross-matching against query video face embedding
        if query_emb is not None and detector is not None and encoder is not None:
            img_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            dets = detector.detect_frame(img_bgr)
            if dets:
                best_det = max(dets, key=lambda d: (d.bbox[2] - d.bbox[0]) * (d.bbox[3] - d.bbox[1]))
                if best_det.landmarks_5 is not None:
                    aligned = align_face_5point(img_bgr, best_det.landmarks_5)
                    if aligned is not None:
                        emb = encoder.extract_embedding(aligned)
                        if emb is not None:
                            sim = float(np.dot(emb, query_emb))
                            if sim > best_sim:
                                best_sim = sim
                                best_match = {
                                    "name": name,
                                    "imdb_id": entry.get("imdb_id", "N/A"),
                                    "popularity": pop,
                                    "image_pil": img,
                                    "cosine_sim": sim
                                }
            count += 1
            if count >= top_k:
                break

    if best_match:
        logger.info(f"Biometric matching identified top candidate: {best_match['name']} (Cosine Sim: {best_match['cosine_sim']:.4f})")
    return best_match


def generate_audit_card(
    enrolled_data: dict,
    video_path: str,
    representative_frame: np.ndarray,
    representative_crop: np.ndarray,
    cosine_sim: float,
    threshold: float,
    is_identity_match: bool,
    manipulation_prob: float,
    save_path: str,
    backbone_label: str = "GenD PE-L"
):
    """Render a 3-panel supervisor diagnostic card visualizing the marketplace decision."""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6.5), dpi=150)

    # Panel 1: Registered Talent Studio Portfolio (TMDB)
    tmdb_img = enrolled_data.get("image_pil")
    ax1.imshow(tmdb_img)
    num_heads = enrolled_data.get("num_headshots", 1)
    portfolio_meta = f"Verified Headshots: {num_heads} (Best Era Match)" if num_heads > 1 else "Single Headshot Enrolled"
    ax1.set_title(f"REGISTERED TALENT PROFILE\nName: {enrolled_data.get('name')}\nIMDb: {enrolled_data.get('imdb_id')}\n{portfolio_meta}", fontsize=11, fontweight="bold", pad=12)
    ax1.axis("off")

    # Panel 2: In-the-Wild Video Observation (Stage 1 & 2)
    frame_rgb = cv2.cvtColor(representative_frame, cv2.COLOR_BGR2RGB)
    ax2.imshow(frame_rgb)
    status_text = "MATCH (TALENT CONFIRMED)" if is_identity_match else "IMPOSTOR / NON-MATCH"
    status_color = "green" if is_identity_match else "red"
    ax2.set_title(f"STAGE 2: BIOMETRIC IDENTITY MATCH\nCosine Similarity: {cosine_sim:.4f} (Threshold: {threshold:.2f})\nStatus: {status_text}", fontsize=11, fontweight="bold", color=status_color, pad=12)
    ax2.axis("off")

    # Panel 3: Forensic Manipulation Assessment (Stage 3 GenD)
    crop_rgb = cv2.cvtColor(representative_crop, cv2.COLOR_BGR2RGB) if representative_crop is not None else frame_rgb
    ax3.imshow(crop_rgb)
    is_fake = manipulation_prob >= 0.50
    verdict_text = "UNAUTHORIZED AI DEEPFAKE" if is_fake else "VERIFIED AUTHENTIC FOOTAGE"
    verdict_color = "red" if is_fake else "green"
    ax3.set_title(f"STAGE 3: FORENSIC VERDICT ({backbone_label})\nProbability p(Fake): {manipulation_prob:.4f}\nVerdict: {verdict_text}", fontsize=11, fontweight="bold", color=verdict_color, pad=12)
    ax3.axis("off")

    # Overall Summary Banner
    banner_color = "#ffebee" if (is_identity_match and is_fake) else "#e8f5e9"
    action_text = "CRITICAL ALERT: Unauthorized deepfake detected of registered talent! Likeness infringement notice issued." if (is_identity_match and is_fake) else ("AUTHORIZED: Real media appearance of enrolled actor." if is_identity_match else "NO ACTION: Video subject does not match registered talent.")
    fig.text(0.5, 0.04, action_text, ha="center", fontsize=12, fontweight="bold", bbox=dict(boxstyle="round,pad=0.6", facecolor=banner_color, edgecolor=verdict_color, lw=2))

    plt.suptitle("TALENT MARKETPLACE AI LIKENESS AUDIT & FORENSIC VERDICT REPORT", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[+] Saved Executive Audit Report Card to: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="End-to-End Talent Marketplace Demo.")
    parser.add_argument("--video-path", type=str, required=True, help="Path to input Celeb-DF video (real or fake).")
    parser.add_argument("--actor-name", type=str, default=None, help="Actor name to search in TMDB (e.g. 'Aamir Khan', 'Tom Cruise').")
    parser.add_argument("--actor-image", type=str, default=None, help="Path to local image of actor for onboarding/enrollment.")
    parser.add_argument("--tmdb-api-key", type=str, default="b2b7416d572f285d036cca47fb16a1c4", help="Official TMDB API Key for multi-headshot portfolio gallery.")
    parser.add_argument("--auto-identify", action="store_true", help="Automatically identify celebrity from TMDB by scanning top actors.")
    parser.add_argument("--forensic-model", type=str, default="vit", choices=["vit", "efficientnet-b4", "sbi", "gend-pe", "gend-clip", "gend-dinov3"], help="Forensic deepfake detection model: 'vit' (ViT Transformer - Recommended for Celeb-DF), 'efficientnet-b4' (SBI Contrastive), 'gend-pe' (GenD Perception Encoder), 'gend-clip' (GenD CLIP-L/14), or 'gend-dinov3' (GenD DINOv3).")
    parser.add_argument("--gend-backbone", type=str, default=None, help="Legacy alias for --forensic-model (e.g. 'pe', 'clip', 'dinov3').")
    parser.add_argument("--output-dir", type=str, default="/kaggle/working/results/marketplace_demo")
    parser.add_argument("--threshold", type=float, default=0.55, help="ArcFace cosine similarity decision threshold.")
    parser.add_argument("--hf-token", type=str, default=None, help="Hugging Face User Access Token (optional if HF_TOKEN env is set).")
    args = parser.parse_args()

    if args.hf_token:
        os.environ["HF_TOKEN"] = args.hf_token

    os.makedirs(args.output_dir, exist_ok=True)

    print("\n" + "="*76)
    print("      TALENT MARKETPLACE: AI LIKENESS MONITORING & FORENSIC AUDIT")
    print("="*76)

    # 1. Initialize Pipeline Engines
    detector = SCRFDDetector(conf_threshold=0.50)
    encoder = ArcFaceEncoder()

    # Resolve forensic detector choice
    model_choice = args.forensic_model.lower()
    if args.gend_backbone is not None:
        model_choice = f"gend-{args.gend_backbone.lower()}"

    if model_choice in ["efficientnet-b4", "sbi", "efficientnet"]:
        logger.info("Initializing Stage 3: SBI Contrastive EfficientNet-B4 ('nikokons/contrastive-deepfake-detector')...")
        forensic_detector = SBIEfficientNetDetector()
        forensic_label = "SBI EfficientNet-B4"
        crop_target_size = (380, 380)
    elif model_choice in ["vit", "vit-base"]:
        logger.info("Initializing Stage 3: ViT Forensics ('dima806/deepfake_vs_real_image_detection')...")
        forensic_detector = ViTDeepfakeDetector()
        forensic_label = "ViT Deepfake Detector"
        crop_target_size = (224, 224)
    elif model_choice in ["gend-pe", "pe"]:
        logger.info("Initializing Stage 3: GenD with Perception Encoder ('yermandy/GenD_PE_L')...")
        forensic_detector = GenDPEDetector()
        forensic_label = "GenD PE-L"
        crop_target_size = (224, 224)
    elif model_choice in ["gend-dinov3", "dinov3"]:
        logger.info("Initializing Stage 3: GenD with Meta DINOv3 ('yermandy/GenD_DINOv3_L')...")
        forensic_detector = GenDDINOv3Detector()
        forensic_label = "GenD DINOv3-L"
        crop_target_size = (224, 224)
    else:
        logger.info("Initializing Stage 3: GenD with OpenAI CLIP-L/14 ('yermandy/GenD_CLIP_L_14')...")
        forensic_detector = GenDCLIPDetector()
        forensic_label = "GenD CLIP-L/14"
        crop_target_size = (224, 224)

    # 2. Extract Primary Actor Face from Video
    logger.info(f"Ingesting monitored video: {args.video_path}")
    reader = VideoReader(args.video_path)
    frames = reader.sample_uniform(max_frames=12)
    if not frames:
        logger.error("Could not read video frames!")
        return

    video_embs = []
    video_crops = []
    representative_frame = None
    representative_crop = None

    for f in frames:
        dets = detector.detect_frame(f.image_bgr, f.frame_index, f.timestamp_sec)
        if dets:
            # Saliency: foreground actor is largest bounding box
            primary = max(dets, key=lambda d: (d.bbox[2] - d.bbox[0]) * (d.bbox[3] - d.bbox[1]))
            if primary.landmarks_5 is not None:
                aligned = align_face_5point(f.image_bgr, primary.landmarks_5)
                if aligned is not None:
                    emb = encoder.extract_embedding(aligned)
                    if emb is not None:
                        video_embs.append(emb)
                        
                        # Model-specific alignment and resolution:
                        # - SBI (EfficientNet-B4): Trained strictly on unwarped bounding box crops with 12.5% margin (380x380)
                        # - ViT: Trained on unwarped bounding box crops (224x224)
                        # - GenD: Trained on 5-point affine landmark alignment (align_face_gend)
                        if model_choice in ["efficientnet-b4", "sbi", "efficientnet"]:
                            forensic_crop = crop_context_face(f.image_bgr, primary.bbox, expansion_factor=1.15, target_size=(380, 380))
                        elif model_choice in ["vit", "vit-base"]:
                            forensic_crop = crop_context_face(f.image_bgr, primary.bbox, expansion_factor=1.20, target_size=(224, 224))
                        else:
                            forensic_crop = align_face_gend(f.image_bgr, primary.landmarks_5, target_size=crop_target_size, scale=1.30)
                            if forensic_crop is None:
                                forensic_crop = crop_context_face(f.image_bgr, primary.bbox, target_size=crop_target_size)

                        if forensic_crop is not None:
                            video_crops.append(forensic_crop)
                            
                        if representative_frame is None:
                            # Annotate representative frame
                            vis_frame = f.image_bgr.copy()
                            bx = primary.bbox.astype(int)
                            cv2.rectangle(vis_frame, (bx[0], bx[1]), (bx[2], bx[3]), (0, 255, 0), 2)
                            representative_frame = vis_frame
                            representative_crop = forensic_crop

    if not video_embs:
        logger.error("No valid faces detected in video!")
        return

    # Video prototype vector
    mean_video_emb = np.mean(np.array(video_embs), axis=0)
    mean_video_emb = mean_video_emb / np.linalg.norm(mean_video_emb)

    # 3. Fetch Registered Actor Portfolio
    enrolled_data = None
    if args.actor_image and os.path.exists(args.actor_image):
        logger.info(f"Enrolling talent from local portfolio image: {args.actor_image}")
        pil_img = Image.open(args.actor_image).convert("RGB")
        name = args.actor_name if args.actor_name else os.path.splitext(os.path.basename(args.actor_image))[0]
        enrolled_data = {
            "name": name,
            "imdb_id": "Local Portfolio",
            "popularity": 100.0,
            "image_pil": pil_img,
            "cosine_sim": 1.0,
            "source": "Local File Enrollment"
        }
    elif args.actor_name:
        # Priority 1: Official TMDB API Multi-Headshot Portfolio Gallery
        if args.tmdb_api_key:
            logger.info(f"Querying Official TMDB API for '{args.actor_name}' multi-image portfolio gallery...")
            enrolled_data = fetch_official_tmdb_talent(
                actor_name=args.actor_name,
                api_key=args.tmdb_api_key,
                detector=detector,
                encoder=encoder,
                query_emb=mean_video_emb
            )
        # Priority 2: Hugging Face Streaming Dataset Fallback
        if enrolled_data is None:
            logger.info("Falling back to Hugging Face 'ashraq/tmdb-people-image' dataset stream...")
            enrolled_data = fetch_tmdb_actor(actor_name=args.actor_name)
    elif args.auto_identify:
        logger.info("Auto-identifying video subject across TMDB popular celebrities...")
        enrolled_data = fetch_tmdb_actor(top_k=80, query_emb=mean_video_emb, encoder=encoder, detector=detector)
        if enrolled_data and args.tmdb_api_key:
            logger.info(f"Auto-identified '{enrolled_data['name']}'. Fetching official TMDB multi-headshot gallery...")
            official_data = fetch_official_tmdb_talent(
                actor_name=enrolled_data["name"],
                api_key=args.tmdb_api_key,
                detector=detector,
                encoder=encoder,
                query_emb=mean_video_emb
            )
            if official_data:
                enrolled_data = official_data
    else:
        # Default fallback: scan top TMDB actors
        enrolled_data = fetch_tmdb_actor(top_k=60, query_emb=mean_video_emb, encoder=encoder, detector=detector)
        if enrolled_data and args.tmdb_api_key:
            official_data = fetch_official_tmdb_talent(
                actor_name=enrolled_data["name"],
                api_key=args.tmdb_api_key,
                detector=detector,
                encoder=encoder,
                query_emb=mean_video_emb
            )
            if official_data:
                enrolled_data = official_data

    if not enrolled_data:
        logger.error("Could not find matching actor profile in TMDB dataset.")
        return

    # 4. Extract Registered Biometric Vector from TMDB Portrait
    if enrolled_data.get("best_emb") is not None:
        enrolled_emb = enrolled_data["best_emb"]
        cosine_sim = float(np.dot(mean_video_emb, enrolled_emb))
    else:
        tmdb_bgr = cv2.cvtColor(np.array(enrolled_data["image_pil"]), cv2.COLOR_RGB2BGR)
        tmdb_dets = detector.detect_frame(tmdb_bgr)
        if not tmdb_dets:
            logger.error("Could not detect face in TMDB portfolio image!")
            return
        best_tmdb_det = max(tmdb_dets, key=lambda d: (d.bbox[2] - d.bbox[0]) * (d.bbox[3] - d.bbox[1]))
        aligned_tmdb = align_face_5point(tmdb_bgr, best_tmdb_det.landmarks_5)
        enrolled_emb = encoder.extract_embedding(aligned_tmdb)
        cosine_sim = float(np.dot(mean_video_emb, enrolled_emb))

    # 5. Stage 2: ArcFace Likeness Verification
    is_match = cosine_sim >= args.threshold

    print("\n" + "-"*76)
    print("STAGE 2: BIOMETRIC LIKENESS VERIFICATION (ARCFACE)")
    print("-" * 76)
    print(f"Registered Talent:            {enrolled_data['name']} (IMDb: {enrolled_data['imdb_id']})")
    print(f"Onboarding Source:            {enrolled_data.get('source', 'TMDB Catalog')}")
    if enrolled_data.get("num_headshots"):
        print(f"Enrolled Gallery Size:        {enrolled_data['num_headshots']} Official Headshots (Multi-Prototype)")
    print(f"Monitored Video File:         {os.path.basename(args.video_path)}")
    print(f"Observed Cosine Similarity:   {cosine_sim:.4f} (Best Portfolio Match)")
    if enrolled_data.get("centroid_emb") is not None:
        centroid_sim = float(np.dot(mean_video_emb, enrolled_data["centroid_emb"]))
        print(f"Centroid Profile Similarity:  {centroid_sim:.4f}")
    print(f"Marketplace Match Threshold:  {args.threshold:.2f}")
    print(f"Identity Verification Status: {'[CONFIRMED MATCH]' if is_match else '[NON-MATCH / IMPOSTOR]'}")

    # 6. Stage 3: Forensic Manipulation Assessment
    manipulation_prob = 0.0
    if video_crops:
        frame_scores = forensic_detector.predict_crops(video_crops)
        manipulation_prob = forensic_detector.aggregate_scores(frame_scores)

    is_fake = manipulation_prob >= 0.50
    print("\n" + "-"*76)
    print(f"STAGE 3: FORENSIC DEEPFAKE DETECTION ({forensic_label})")
    print("-" * 76)
    print(f"Manipulation Score p(Fake):   {manipulation_prob:.4f}")
    print(f"Forensic Verdict:             {'[UNAUTHORIZED DEEPFAKE INFRINGEMENT]' if is_fake else '[VERIFIED AUTHENTIC FOOTAGE]'}")
    print("="*76 + "\n")

    # 7. Generate 3-Panel Executive Diagnostic Report Card
    report_path = os.path.join(args.output_dir, f"marketplace_audit_{os.path.splitext(os.path.basename(args.video_path))[0]}.png")
    generate_audit_card(
        enrolled_data=enrolled_data,
        video_path=args.video_path,
        representative_frame=representative_frame,
        representative_crop=representative_crop,
        cosine_sim=cosine_sim,
        threshold=args.threshold,
        is_identity_match=is_match,
        manipulation_prob=manipulation_prob,
        save_path=report_path,
        backbone_label=forensic_label
    )


if __name__ == "__main__":
    main()
