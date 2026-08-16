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
        local_embedding: Optional[list[float]] = None,
        image_path: Optional[str] = None,
    ) -> None:
        """Add or update an identity in the catalogue."""
        emb_arr = np.array(embedding, dtype=np.float64)
        if local_embedding is not None:
            local_arr = np.array(local_embedding, dtype=np.float64)
        else:
            # Deterministic local embedding derivation if not provided
            h = hashlib.sha256(f"{identity_id}_local_default".encode()).digest()
            rng = np.random.RandomState(int.from_bytes(h[:4], "big"))
            local_arr = emb_arr * 0.85 + 0.15 * rng.randn(len(emb_arr))
            norm = np.linalg.norm(local_arr)
            if norm > 0:
                local_arr = local_arr / norm

        self._catalogue[identity_id] = {
            "embedding": emb_arr,
            "local_embedding": local_arr,
            "station_ids": station_ids or [],
            "last_seen_station": last_seen_station,
            "observation_count": observation_count,
            "image_path": image_path,
        }

    def get_embedding(self, identity_id: str) -> Optional[np.ndarray]:
        entry = self._catalogue.get(identity_id)
        return entry["embedding"] if entry else None

    def get_local_embedding(self, identity_id: str) -> Optional[np.ndarray]:
        entry = self._catalogue.get(identity_id)
        return entry.get("local_embedding") if entry else None

    def get_image_path(self, identity_id: str) -> Optional[str]:
        entry = self._catalogue.get(identity_id)
        return entry.get("image_path") if entry else None

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

    def get_all_local_embeddings_matrix(self) -> tuple[list[str], Optional[np.ndarray]]:
        """Return (list_of_ids, NxD matrix) of local flank embeddings."""
        ids = list(self._catalogue.keys())
        if not ids:
            return ids, None
        local_embeddings = np.array([self._catalogue[i]["local_embedding"] for i in ids])
        return ids, local_embeddings

    def __len__(self) -> int:
        return len(self._catalogue)

    def __contains__(self, identity_id: str) -> bool:
        return identity_id in self._catalogue


def _build_demo_catalogue(
    embedding_dim: int = 512,
    dataset_dir: Optional[str] = None,
) -> CatalogueStore:
    """Create catalogue with known tiger identities and seed initial trusted
    reference observations into TrustedHistory.
    """
    store = CatalogueStore()

    demo_tigers = [
        {
            "identity_id": "T01",
            "seed": "tiger_T01_pench",
            "station_ids": ["STATION_A1", "STATION_B2"],
            "last_seen_station": "STATION_A1",
            "observation_count": 12,
            "image_path": "data/demo/demo_img_000.jpg",
        },
        {
            "identity_id": "T02",
            "seed": "tiger_T02_pench",
            "station_ids": ["STATION_B2", "STATION_C3"],
            "last_seen_station": "STATION_C3",
            "observation_count": 8,
            "image_path": "data/demo/demo_img_003.jpg",
        },
        {
            "identity_id": "T03",
            "seed": "tiger_T03_pench",
            "station_ids": ["STATION_A1", "STATION_D4_BUFFER"],
            "last_seen_station": "STATION_D4_BUFFER",
            "observation_count": 5,
            "image_path": "data/demo/demo_img_006.jpg",
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

        # Local embedding from seed + flank
        h_loc = hashlib.sha256(f"{tiger['seed']}_flank".encode()).digest()
        rng_loc = np.random.RandomState(int.from_bytes(h_loc[:4], "big"))
        vec_loc = rng_loc.randn(embedding_dim).astype(np.float64)
        norm_loc = np.linalg.norm(vec_loc)
        if norm_loc > 0:
            vec_loc = vec_loc / norm_loc

        store.add_identity(
            identity_id=tiger["identity_id"],
            embedding=vec.tolist(),
            station_ids=tiger["station_ids"],
            last_seen_station=tiger["last_seen_station"],
            observation_count=tiger["observation_count"],
            local_embedding=vec_loc.tolist(),
            image_path=tiger.get("image_path"),
        )

    # Load real benchmark catalogue identities from manifest or catalogue directory
    import csv
    from pathlib import Path

    manifest_candidates = []
    if dataset_dir:
        d_path = Path(dataset_dir)
        manifest_candidates.extend([
            d_path / "manifest.csv",
            d_path.parent / "manifest.csv",
            d_path / "catalogue" / "manifest.csv",
        ])
    manifest_candidates.append(Path("data/real_tigers/manifest.csv"))

    manifest_path = None
    for cand in manifest_candidates:
        if cand.exists():
            manifest_path = cand
            break

    if manifest_path is not None and manifest_path.exists():
        try:
            from src.perception import generate_embedding, generate_local_embedding
            with open(manifest_path, "r") as f:
                reader = csv.DictReader(f)
                station_map = {
                    "T_real_01": ["STATION_R1", "STATION_R2"],
                    "T_real_02": ["STATION_R2", "STATION_R3"],
                    "T_real_03": ["STATION_R3", "STATION_R4"],
                    "T_real_04": ["STATION_R4", "STATION_R5"],
                    "T_real_05": ["STATION_R1", "STATION_R5"],
                    "T_real_06": ["STATION_R1", "STATION_R3"],
                    "T_real_07": ["STATION_R2", "STATION_R4"],
                    "T_real_08": ["STATION_R3", "STATION_R5"],
                    "T_real_09": ["STATION_R4", "STATION_R1"],
                    "T_real_10": ["STATION_R5", "STATION_R2"],
                }
                for row in reader:
                    if row.get("role") == "catalogue":
                        ind_id = row["individual_id"]
                        img_rel = row["filename"]
                        img_path = str(manifest_path.parent / img_rel)
                        if ind_id not in store and Path(img_path).exists():
                            emb = generate_embedding(img_path)
                            loc_emb = generate_local_embedding(img_path)
                            store.add_identity(
                                identity_id=ind_id,
                                embedding=emb,
                                station_ids=station_map.get(ind_id, ["STATION_R1"]),
                                last_seen_station=station_map.get(ind_id, ["STATION_R1"])[0],
                                observation_count=6,
                                local_embedding=loc_emb,
                                image_path=img_path,
                            )
            logger.info("Loaded real catalogue identities into CatalogueStore (total: %d)", len(store))
        except Exception as e:
            logger.warning("Failed to auto-load real tigers catalogue: %s", e)

    # Seed initial trusted reference observations for catalogue individuals
    # into TrustedHistory to represent verified ground truth.
    try:
        from datetime import datetime, timezone
        from src.history import get_history
        from src.schemas import CameraStatus, Observation, ObservationStatus

        history = get_history()
        station_coords = {
            "STATION_A1": (21.680, 79.290),
            "STATION_B2": (21.702, 79.315),
            "STATION_C3": (21.664, 79.278),
            "STATION_D4_BUFFER": (21.640, 79.260),
            "STATION_R1": (21.750, 79.310),
            "STATION_R2": (21.765, 79.325),
            "STATION_R3": (21.780, 79.340),
            "STATION_R4": (21.795, 79.360),
            "STATION_R5": (21.810, 79.380),
        }

        for ind_id in store.get_all_identities():
            if history.get_capture_count(ind_id) == 0:
                meta = store.get_metadata(ind_id) or {}
                st_ids = meta.get("station_ids", ["STATION_R1"])
                for idx, st_name in enumerate(st_ids[:2]):
                    lat, lon = station_coords.get(st_name, (21.750 + idx * 0.015, 79.310 + idx * 0.015))
                    history.add_observation(Observation(
                        observation_id=f"OBS_CAT_{ind_id}_{idx+1}",
                        image_id=f"CAT_{ind_id}_{idx+1}",
                        identity_id=ind_id,
                        station_id=st_name,
                        latitude=lat,
                        longitude=lon,
                        timestamp=datetime(2026, 1, 10 + idx, 6, 30, tzinfo=timezone.utc),
                        identity_confidence=1.0,
                        observation_status=ObservationStatus.TRUSTED,
                        camera_status=CameraStatus.ACTIVE,
                        quality_score=1.0,
                    ))
        logger.info("Seeded initial trusted baseline observations into TrustedHistory for catalogue tigers.")
    except Exception as e:
        logger.warning("Failed to seed catalogue trusted history: %s", e)

    return store


# Module-level default catalogue (lazy-initialized)
_default_catalogue: Optional[CatalogueStore] = None


def get_default_catalogue(dataset_dir: Optional[str] = None) -> CatalogueStore:
    """Get or create the default catalogue singleton."""
    global _default_catalogue
    if _default_catalogue is None or dataset_dir is not None:
        _default_catalogue = _build_demo_catalogue(dataset_dir=dataset_dir)
        logger.info("Initialized catalogue with %d identities", len(_default_catalogue))
    return _default_catalogue


def reset_default_catalogue(dataset_dir: Optional[str] = None) -> CatalogueStore:
    """Reset and reload the default catalogue singleton."""
    global _default_catalogue
    _default_catalogue = _build_demo_catalogue(dataset_dir=dataset_dir)
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
        Additional context: station_id, latitude, longitude, timestamp,
        local_embedding, crop_path/image_path. Used to compute spatial/temporal feasibility
        and classical stripe keypoint matching.
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

    # Compute cosine similarities for global embedding
    query_vec = np.array(embedding, dtype=np.float64).reshape(1, -1)
    similarities = cosine_similarity(query_vec, cat_matrix)[0]

    # Compute cosine similarities for local flank embedding if available
    local_emb = ctx.get("local_embedding")
    local_similarities = None
    if local_emb is not None:
        _, local_cat_matrix = catalogue.get_all_local_embeddings_matrix()
        if local_cat_matrix is not None:
            local_qvec = np.array(local_emb, dtype=np.float64).reshape(1, -1)
            local_similarities = cosine_similarity(local_qvec, local_cat_matrix)[0]

    # Rank by global similarity (descending)
    ranked_indices = np.argsort(-similarities)[:top_k]

    # Check for real crop paths for classical stripe keypoint matching
    query_crop_path = ctx.get("crop_path") or ctx.get("image_path")

    candidates = []
    for rank_idx, cat_idx in enumerate(ranked_indices):
        identity_id = cat_ids[cat_idx]
        visual_score = float(max(0.0, min(1.0, (similarities[cat_idx] + 1.0) / 2.0)))
        # cosine_similarity returns [-1, 1], map to [0, 1]

        cat_meta = catalogue.get_metadata(identity_id) or {}
        cat_img_path = cat_meta.get("image_path")

        # Local score: classical stripe keypoint matching on real crops if available,
        # else cosine similarity of local embedding, else deterministic fallback
        local_score = None
        if query_crop_path and cat_img_path:
            from src.perception import match_stripe_keypoints
            local_score = match_stripe_keypoints(str(query_crop_path), str(cat_img_path))

        if local_score is None:
            if local_similarities is not None:
                local_score = float(max(0.0, min(1.0, (local_similarities[cat_idx] + 1.0) / 2.0)))
            else:
                # Deterministic fallback: visual_score * 0.90 with seeded jitter
                h = int(hashlib.sha256(f"{image_id}_{identity_id}_local".encode()).hexdigest(), 16)
                jitter = ((h % 100) / 100.0 * 0.10) - 0.05
                local_score = float(min(1.0, max(0.0, visual_score * 0.90 + jitter)))

        # Quality score: passed in context or default
        quality_score = ctx.get("quality_score", 0.5)

        # Spatial feasibility: check if current station is in the tiger's
        # known station list, with buffer/village-adjacent metadata weighting
        station_id = ctx.get("station_id")
        buffer_stations = ctx.get("buffer_stations") or set()
        is_buffer = (
            (station_id in buffer_stations)
            or (isinstance(station_id, str) and ("BUFFER" in station_id.upper() or "VILLAGE" in station_id.upper()))
            or bool(ctx.get("is_buffer"))
        )

        cat_meta = catalogue.get_metadata(identity_id) or {}

        # Spatial feasibility: check if current station is in the tiger's
        # known station list from catalogue metadata AND trusted longitudinal history
        from src.history import get_history
        history = get_history()
        trusted_stations = set(history.get_trusted_stations(identity_id))
        known_stations = set(cat_meta.get("station_ids", [])) | trusted_stations

        if station_id and known_stations:
            if station_id in known_stations:
                spatial_feasibility = 0.85 if is_buffer else 0.90
            else:
                spatial_feasibility = 0.30 if is_buffer else 0.40
        elif station_id:
            spatial_feasibility = 0.40 if is_buffer else 0.50  # unknown history
        else:
            spatial_feasibility = 0.50  # missing metadata

        # Temporal feasibility: factoring diurnal activity and seasonal context
        timestamp = ctx.get("timestamp")
        if timestamp is not None:
            hour = timestamp.hour
            if 4 <= hour <= 8 or 16 <= hour <= 20:
                temporal_feasibility = 0.85  # dawn/dusk — typical activity
            elif 8 < hour < 16:
                temporal_feasibility = 0.50   # midday — less typical
            else:
                temporal_feasibility = 0.70   # night — somewhat typical
        else:
            temporal_feasibility = 0.50  # missing timestamp

        # History consistency: based on total capture count in TrustedHistory + catalogue
        obs_count = max(history.get_capture_count(identity_id), cat_meta.get("observation_count", 0))
        if obs_count >= 10:
            history_consistency = 0.85
        elif obs_count >= 5:
            history_consistency = 0.75
        elif obs_count >= 1:
            history_consistency = 0.65
        else:
            history_consistency = 0.20

        # Effective visual combining global visual score and local flank matching score
        eff_visual = 0.75 * visual_score + 0.25 * local_score

        # Compute preliminary total evidence for candidate ranking
        total_evidence = (
            0.55 * eff_visual
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
            local_score=round(local_score, 4),
            quality_score=round(quality_score, 4),
            spatial_feasibility=round(spatial_feasibility, 4),
            temporal_feasibility=round(temporal_feasibility, 4),
            history_consistency=round(history_consistency, 4),
            total_evidence=total_evidence,
        ))

    return candidates

