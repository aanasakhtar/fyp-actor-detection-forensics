"""
Enrolled actor reference gallery and embedding management.
"""

import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from ..utils.logging import get_logger

logger = get_logger("reference_gallery")

@dataclass
class EnrolledActor:
    """Represents an enrolled registered actor in the platform gallery."""
    actor_id: str
    name: str
    reference_embeddings: List[np.ndarray] = field(default_factory=list)
    prototype_embedding: Optional[np.ndarray] = None
    reference_crops: List[np.ndarray] = field(default_factory=list)

class ActorReferenceGallery:
    """Manages enrolled actor profiles and prototype biometric embeddings."""
    
    def __init__(self):
        self.actors: Dict[str, EnrolledActor] = {}

    @property
    def prototypes(self) -> Dict[str, np.ndarray]:
        """Dictionary mapping enrolled actor_id to their prototype embedding vector."""
        return {
            actor_id: actor.prototype_embedding
            for actor_id, actor in self.actors.items()
            if actor.prototype_embedding is not None
        }

    @property
    def actor_ids(self) -> List[str]:
        """List of all enrolled actor IDs."""
        return list(self.actors.keys())
        
    def enroll_actor(
        self, 
        actor_id: str, 
        name: str, 
        embeddings: List[np.ndarray],
        crops: Optional[List[np.ndarray]] = None
    ) -> None:
        """
        Enroll or update an actor with multiple reference facial embeddings.
        Computes the L2-normalized mean prototype vector.
        """
        valid_embs = [e for e in embeddings if e is not None and len(e) > 0]
        if not valid_embs:
            logger.warning(f"No valid embeddings provided for actor {actor_id}")
            return
            
        # Compute mean prototype vector
        stacked = np.vstack(valid_embs)
        mean_vec = np.mean(stacked, axis=0)
        norm = np.linalg.norm(mean_vec)
        if norm > 0:
            proto = (mean_vec / norm).astype(np.float32)
        else:
            proto = mean_vec.astype(np.float32)
            
        valid_crops = crops if crops is not None else []
        self.actors[actor_id] = EnrolledActor(
            actor_id=actor_id,
            name=name,
            reference_embeddings=valid_embs,
            prototype_embedding=proto,
            reference_crops=valid_crops
        )
        logger.info(f"Enrolled actor '{name}' ({actor_id}) with {len(valid_embs)} reference images.")

    def get_reference_crop(self, actor_id: str) -> Optional[np.ndarray]:
        """Retrieve the primary reference face crop for visualization."""
        actor = self.actors.get(actor_id)
        if actor and actor.reference_crops:
            return actor.reference_crops[0]
        return None

    def remove_actor(self, actor_id: str) -> bool:
        """Remove an actor from the gallery upon consent withdrawal."""
        if actor_id in self.actors:
            del self.actors[actor_id]
            logger.info(f"Removed actor {actor_id} from gallery.")
            return True
        return False

    def get_prototype(self, actor_id: str) -> Optional[np.ndarray]:
        """Retrieve prototype embedding for an actor."""
        actor = self.actors.get(actor_id)
        return actor.prototype_embedding if actor else None

    def __len__(self) -> int:
        return len(self.actors)
