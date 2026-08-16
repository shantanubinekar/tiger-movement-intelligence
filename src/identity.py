"""
identity.py — Candidate identity generation (ranked matching).

Developer 2 owns this file (PROJECT_CONTRACT.md Section 18).

Responsibilities:
- generate_candidates(embedding, catalogue) -> list[IdentityCandidate]
  - Use cosine similarity for nearest-neighbor lookup.
  - Return top-3 candidates plus the top-1/top-2 margin.
  - This function MUST NOT decide trusted/ambiguous/unknown — it only
    produces ranked evidence.
- CatalogueStore: in-memory catalogue of known tiger identities.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Optional

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from src.schemas import IdentityCandidate, ImageRecord

logger = logging.getLogger(__name__)


class CatalogueStore:
    """In-memory catalogue of known tiger identities.

    Stores identity_id -> {embedding, metadata} mapping.
    In Phase 1, pre-populated with demo/synthetic identities.
    """

    def __init__(self):
        self._catalogue: dict[str, dict] = {}

    def add_identity(
        self,
        identity_id: str,
        embedding: list[float],
        station_ids: Optional[list[str]] = None,
        last_seen_station: Optional[str] = None,
        observation_count: int = 0,
    ) -> None:
        """Add or update an identity in the catalogue."""
        self._catalogue[identity_id] = {
            "embedding": np.array(embedding, dtype=np.float64),
            "station_ids": station_ids or [],
            "last_seen_station": last_seen_station,
            "observation_count": observation_count,
        }

    def get_embedding(self, identity_id: str) -> Optional[np.ndarray]:
        entry = self._catalogue.get(identity_id)
        return entry["embedding"] if entry else None

    def get_all_identities(self) -> list[str]:
        return list(self._catalogue.keys())

    def get_metadata(self, identity_id: str) -> Optional[dict]:
        return self._catalogue.get(identity_id)

    def get_all_embeddings_matrix(self) -> tuple[list[str], Optional[np.ndarray]]:
        """Return (list_of_ids, NxD matrix) for bulk similarity computation."""
        ids = list(self._catalogue.keys())
        if not ids:
            return ids, None
        embeddings = np.array([self._catalogue[i]["embedding"] for i in ids])
        return ids, embeddings

    def __len__(self) -> int:
        return len(self._catalogue)

    def __contains__(self, identity_id: str) -> bool:
        return identity_id in self._catalogue


def _build_demo_catalogue(embedding_dim: int = 512) -> CatalogueStore:
    """Create a demo catalogue with 3 synthetic tiger identities.
    Embeddings are deterministic and reproducible."""
    store = CatalogueStore()

    demo_tigers = [
        {
            "identity_id": "T01",
            "seed": "tiger_T01_pench",
            "station_ids": ["STATION_A1", "STATION_B2"],
            "last_seen_station": "STATION_A1",
            "observation_count": 12,
        },
        {
            "identity_id": "T02",
            "seed": "tiger_T02_pench",
            "station_ids": ["STATION_B2", "STATION_C3"],
            "last_seen_station": "STATION_C3",
            "observation_count": 8,
        },
        {
            "identity_id": "T03",
            "seed": "tiger_T03_pench",
            "station_ids": ["STATION_A1", "STATION_D4_BUFFER"],
            "last_seen_station": "STATION_D4_BUFFER",
            "observation_count": 5,
        },
    ]

    for tiger in demo_tigers:
        # Deterministic embedding from seed
        h = hashlib.sha256(tiger["seed"].encode()).digest()
        rng = np.random.RandomState(int.from_bytes(h[:4], "big"))
        vec = rng.randn(embedding_dim).astype(np.float64)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        store.add_identity(
            identity_id=tiger["identity_id"],
            embedding=vec.tolist(),
            station_ids=tiger["station_ids"],
            last_seen_station=tiger["last_seen_station"],
            observation_count=tiger["observation_count"],
        )

    return store


# Module-level default catalogue (lazy-initialized)
_default_catalogue: Optional[CatalogueStore] = None


def get_default_catalogue() -> CatalogueStore:
    """Get or create the default demo catalogue."""
    global _default_catalogue
    if _default_catalogue is None:
        _default_catalogue = _build_demo_catalogue()
        logger.info("Initialized demo catalogue with %d identities", len(_default_catalogue))
    return _default_catalogue


def generate_candidates(
    embedding: list[float],
    image_id: str,
    catalogue: Optional[CatalogueStore] = None,
    context: Optional[dict] = None,
    top_k: int = 3,
) -> list[IdentityCandidate]:
    """Given an embedding vector and a catalogue, return the top-K most
    similar identity candidates ranked by total evidence.

    This function does NOT decide trusted/ambiguous/unknown — it only
    produces ranked evidence for downstream gating.

    Parameters
    ----------
    embedding : list[float]
        Query embedding vector (unit-normalized).
    image_id : str
        ID of the image being matched.
    catalogue : CatalogueStore, optional
        Identity catalogue to search. Uses demo catalogue if None.
    context : dict, optional
        Additional context: station_id, latitude, longitude, timestamp.
        Used to compute spatial/temporal feasibility.
    top_k : int
        Number of top candidates to return.

    Returns
    -------
    list[IdentityCandidate]
        Ranked candidates with component evidence scores.
    """
    if catalogue is None:
        catalogue = get_default_catalogue()

    if len(catalogue) == 0:
        logger.warning("Empty catalogue — returning empty candidate list")
        return []

    ctx = context or {}

    # Get all catalogue embeddings
    cat_ids, cat_matrix = catalogue.get_all_embeddings_matrix()
    if cat_matrix is None:
        return []

    # Compute cosine similarities
    query_vec = np.array(embedding, dtype=np.float64).reshape(1, -1)
    similarities = cosine_similarity(query_vec, cat_matrix)[0]

    # Rank by similarity (descending)
    ranked_indices = np.argsort(-similarities)[:top_k]

    candidates = []
    for rank_idx, cat_idx in enumerate(ranked_indices):
        identity_id = cat_ids[cat_idx]
        visual_score = float(max(0.0, min(1.0, (similarities[cat_idx] + 1.0) / 2.0)))
        # cosine_similarity returns [-1, 1], map to [0, 1]

        # Quality score: passed in context or default
        quality_score = ctx.get("quality_score", 0.5)

        # Spatial feasibility: check if current station is in the tiger's
        # known station list
        station_id = ctx.get("station_id")
        cat_meta = catalogue.get_metadata(identity_id) or {}
        known_stations = cat_meta.get("station_ids", [])
        if station_id and known_stations:
            spatial_feasibility = 0.9 if station_id in known_stations else 0.4
        elif station_id:
            spatial_feasibility = 0.5  # unknown history
        else:
            spatial_feasibility = 0.5  # missing metadata

        # Temporal feasibility: for Phase 1, use a simple proxy —
        # tigers are more active at dawn/dusk. More sophisticated
        # temporal analysis would go in a P1/P2 iteration.
        timestamp = ctx.get("timestamp")
        if timestamp is not None:
            hour = timestamp.hour
            if 4 <= hour <= 8 or 16 <= hour <= 20:
                temporal_feasibility = 0.85  # dawn/dusk — typical activity
            elif 8 < hour < 16:
                temporal_feasibility = 0.5   # midday — less typical
            else:
                temporal_feasibility = 0.7   # night — somewhat typical
        else:
            temporal_feasibility = 0.5  # missing timestamp

        # History consistency: based on observation count in catalogue
        obs_count = cat_meta.get("observation_count", 0)
        if obs_count >= 10:
            history_consistency = 0.85
        elif obs_count >= 5:
            history_consistency = 0.65
        elif obs_count >= 1:
            history_consistency = 0.45
        else:
            history_consistency = 0.2

        # Compute total evidence using the prototype formula
        # E = 0.55*V + 0.15*Q + 0.15*S + 0.10*T + 0.05*H
        # NOTE: this is duplicated in gating.py with configurable weights.
        # Here we compute a preliminary total for ranking purposes.
        total_evidence = (
            0.55 * visual_score
            + 0.15 * quality_score
            + 0.15 * spatial_feasibility
            + 0.10 * temporal_feasibility
            + 0.05 * history_consistency
        )
        total_evidence = round(min(1.0, max(0.0, total_evidence)), 4)

        candidates.append(IdentityCandidate(
            image_id=image_id,
            candidate_identity=identity_id,
            rank=rank_idx + 1,
            visual_score=round(visual_score, 4),
            quality_score=round(quality_score, 4),
            spatial_feasibility=round(spatial_feasibility, 4),
            temporal_feasibility=round(temporal_feasibility, 4),
            history_consistency=round(history_consistency, 4),
            total_evidence=total_evidence,
        ))

    return candidates
