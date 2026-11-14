"""
Actor verification and cosine similarity matching logic.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, List
from .reference_gallery import ActorReferenceGallery

@dataclass
class MatchResult:
    """Result of comparing a detected face against the enrolled gallery."""
    matched_actor_id: Optional[str]
    matched_actor_name: Optional[str]
    similarity_score: float
    status: str  # 'possible match', 'uncertain', 'unknown'

class IdentityMatcher:
    """Matches facial embeddings against an ActorReferenceGallery using cosine similarity."""
    
    def __init__(
        self, 
        gallery: ActorReferenceGallery, 
        match_threshold: float = 0.45, 
        uncertain_threshold: float = 0.35
    ):
        self.gallery = gallery
        self.match_threshold = match_threshold
        self.uncertain_threshold = uncertain_threshold
        
    def match(self, embedding: np.ndarray) -> MatchResult:
        """
        Compare query embedding against all enrolled actors in the gallery.
        Returns the closest matching enrolled actor and confidence category.
        """
        if embedding is None or len(self.gallery) == 0:
            return MatchResult(None, None, 0.0, "unknown")
            
        # Ensure query is normalized
        norm = np.linalg.norm(embedding)
        if norm > 0:
            q = embedding / norm
        else:
            q = embedding
            
        best_actor_id = None
        best_name = None
        best_sim = -1.0
        
        for actor_id, actor in self.gallery.actors.items():
            if actor.prototype_embedding is None:
                continue
            sim = float(np.dot(q, actor.prototype_embedding))
            if sim > best_sim:
                best_sim = sim
                best_actor_id = actor_id
                best_name = actor.name
                
        # Determine status according to thresholds
        if best_sim >= self.match_threshold:
            status = "possible match"
        elif best_sim >= self.uncertain_threshold:
            status = "uncertain"
        else:
            status = "unknown"
            best_actor_id = None
            best_name = None
            
        return MatchResult(
            matched_actor_id=best_actor_id,
            matched_actor_name=best_name,
            similarity_score=max(0.0, best_sim),
            status=status
        )
