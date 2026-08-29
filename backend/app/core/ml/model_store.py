"""
model_store.py — In-process AXE model variant store.
Loaded once at startup; zero disk I/O at inference time.
"""
import json
import logging
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)

class ModelStore:
    """
    Singleton registry of loaded Keras model instances.
    Key format: "{model_id}:{variant_tag}"  e.g. "axe_genesis_v2:market"
    """

    def __init__(self) -> None:
        # We don't import keras immediately to keep startup fast,
        # but the models will be loaded here.
        self._store: Dict[str, 'keras.Model'] = {}
        self._meta:  Dict[str, dict]        = {}

    # ── Load ──────────────────────────────────────────────────────────────────
    def load_variant(self, manifest: dict, weights_path: Path) -> str:
        """Build model graph, load weights, cache — returns composite key."""
        key = f"{manifest['model_id']}:{manifest.get('variant_tag', 'market')}"
        if key in self._store:
            logger.info("[ModelStore] Already loaded: %s — skipping", key)
            return key

        model = self._build_model(manifest, weights_path)
        self._store[key] = model
        self._meta[key]  = manifest
        
        # Determine params without importing tensorflow at the top
        params = getattr(model, "count_params", lambda: 0)()
        if params == 0 and hasattr(model, "context_model"):
            params = model.context_model.count_params()
        logger.info("[ModelStore] ✅ Loaded %s (%s params)", key, f"{params:,}")
        return key

    def _build_model(self, manifest: dict, weights_path: Path):
        """Delegate to the axe_genesis package builder."""
        import sys
        backend_root = Path(__file__).resolve().parents[3]
        if str(backend_root) not in sys.path:
            sys.path.insert(0, str(backend_root))
            
        model_id = manifest.get("model_id")
        if model_id == "axe_genesis_v2":
            from app.core.ml.axe_genesis_v2_runtime import AXEGenesisV2Runtime
            from app.core.ml.inference_feature_pipeline import resolve_dataset_cache_dir
            checkpoint_dir = weights_path.parent
            dataset_name = manifest.get("dataset_name")
            dataset_dir = resolve_dataset_cache_dir(dataset_name)
            variant_tag = manifest.get("variant_tag", "market")
            logger.info("[ModelStore] Initializing AXEGenesisV2Runtime for variant=%s", variant_tag)
            return AXEGenesisV2Runtime(checkpoint_dir=checkpoint_dir, dataset_dir=dataset_dir, variant_tag=variant_tag)
            
        import axe_genesis
        builder_name = manifest.get("builder", "build_context_model")
        builder_fn = getattr(axe_genesis, builder_name)
        input_shape = tuple(manifest["input_shape"])
        # Pass n_features so builder can be called consistently
        n_features = manifest.get("n_features", input_shape[-1])
        model = builder_fn(input_shape=(input_shape[0], n_features))
        model.load_weights(str(weights_path))
        return model

    # ── Retrieve ──────────────────────────────────────────────────────────────
    def get(self, model_id: str, variant_tag: str = "market"):
        return self._store.get(f"{model_id}:{variant_tag}")

    def get_by_key(self, key: str):
        return self._store.get(key)
        
    def get_meta(self, model_id: str, variant_tag: str = "market"):
        return self._meta.get(f"{model_id}:{variant_tag}")

    # ── List ──────────────────────────────────────────────────────────────────
    def list_loaded(self) -> list[dict]:
        return [
            {**meta, "key": key, "params": getattr(self._store[key], "count_params", lambda: 0)()}
            for key, meta in self._meta.items()
        ]

    # ── Scan & bulk load ─────────────────────────────────────────────────────
    def load_all_variants(self, weights_root: Path) -> int:
        """
        Scan weights_root for *.variant.json manifests and load each one.
        Called once from FastAPI lifespan startup.
        """
        manifests = list(weights_root.rglob("*.variant.json"))
        logger.info("[ModelStore] Found %d variant manifests in %s", len(manifests), weights_root)
        loaded = 0
        
        # To persist these into the DB, we could either do it here or in the caller.
        # It's cleaner to just load models into memory here, and then do a DB sync loop.
        db_variants_to_sync = []
        
        for mf in sorted(manifests):
            try:
                manifest = json.loads(mf.read_text())
                weights_file = mf.parent / manifest["weights_file"]
                if not weights_file.exists():
                    logger.warning("[ModelStore] Weights not found for %s (%s) — skipping", mf.name, weights_file)
                    continue
                self.load_variant(manifest, weights_file)
                
                # Attach the weights_path relative to weights_root for DB
                manifest["weights_path"] = str(weights_file.relative_to(weights_root))
                db_variants_to_sync.append(manifest)
                loaded += 1
            except Exception as exc:
                logger.error("[ModelStore] Failed to load %s: %s", mf.name, exc, exc_info=True)
                
        self._sync_variants_to_db_sync(db_variants_to_sync)
        logger.info("[ModelStore] Startup complete — %d/%d variants loaded", loaded, len(manifests))
        return loaded
        
    def _sync_variants_to_db_sync(self, manifests: list[dict]):
        """Synchronous DB sync because this is called in a thread pool executor."""
        try:
            from app.api.routes.data.database import engine
            from sqlalchemy.orm import sessionmaker
            from sqlalchemy import text
            
            SessionLocal = sessionmaker(bind=engine)
            with SessionLocal() as db:
                for mf in manifests:
                    # Upsert variant in DB
                    stmt = text("""
                        INSERT INTO axe_model_variants 
                        (model_id, variant_tag, display_name, feature_set, n_features, input_shape, dataset_name, weights_path, tier)
                        VALUES (:model_id, :variant_tag, :display_name, :feature_set, :n_features, :input_shape, :dataset_name, :weights_path, :tier)
                        ON CONFLICT (model_id, variant_tag) DO UPDATE SET
                            display_name = EXCLUDED.display_name,
                            feature_set = EXCLUDED.feature_set,
                            n_features = EXCLUDED.n_features,
                            input_shape = EXCLUDED.input_shape,
                            dataset_name = EXCLUDED.dataset_name,
                            weights_path = EXCLUDED.weights_path,
                            tier = EXCLUDED.tier,
                            is_active = true
                    """)
                    db.execute(stmt, {
                        "model_id": mf["model_id"],
                        "variant_tag": mf.get("variant_tag", "market"),
                        "display_name": mf.get("display_name"),
                        "feature_set": mf.get("feature_set"),
                        "n_features": mf.get("n_features"),
                        "input_shape": json.dumps(mf.get("input_shape", [])),
                        "dataset_name": mf.get("dataset_name"),
                        "weights_path": mf.get("weights_path"),
                        "tier": mf.get("tier", "production")
                    })
                db.commit()
                logger.info("[ModelStore] Synced %d variants to database.", len(manifests))
        except Exception as e:
            logger.error("[ModelStore] Failed to sync variants to database: %s", e)


# Module-level singleton — imported by routers and lifespan
model_store = ModelStore()
