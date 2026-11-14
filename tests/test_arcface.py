"""
Unit tests for ArcFace reference gallery, embedding normalization, and identity matching.
"""

import numpy as np
from pipeline.recognition.reference_gallery import ActorReferenceGallery
from pipeline.recognition.identity_matching import IdentityMatcher

def test_gallery_enrollment_and_prototype():
    gallery = ActorReferenceGallery()
    emb1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    emb2 = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    
    gallery.enroll_actor("actor_1", "Actor One", [emb1, emb2])
    proto = gallery.get_prototype("actor_1")
    
    assert proto is not None
    # L2 norm must equal 1.0
    assert abs(np.linalg.norm(proto) - 1.0) < 1e-5

def test_identity_matching_thresholds():
    gallery = ActorReferenceGallery()
    proto = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    gallery.enroll_actor("actor_1", "Actor One", [proto])
    
    matcher = IdentityMatcher(gallery, match_threshold=0.70, uncertain_threshold=0.40)
    
    # Exact match -> similarity 1.0
    res_exact = matcher.match(np.array([1.0, 0.0, 0.0], dtype=np.float32))
    assert res_exact.status == "possible match"
    assert res_exact.matched_actor_id == "actor_1"
    
    # Uncertain match -> similarity ~0.50
    vec_uncertain = np.array([0.5, np.sqrt(0.75), 0.0], dtype=np.float32)
    res_uncertain = matcher.match(vec_uncertain)
    assert res_uncertain.status == "uncertain"
    
    # Unknown match -> orthogonal vector
    vec_unknown = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    res_unknown = matcher.match(vec_unknown)
    assert res_unknown.status == "unknown"
