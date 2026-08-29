import asyncio
import logging
import uuid
import json
from typing import Any, Dict, Optional

import pandas as pd

from app.core.services.websocket_manager import get_websocket_manager
from app.core.ml.proprietary_model_runtime import proprietary_model_runtime, ProprietaryModelContractError
from app.core.cache.redis_client import get_redis

logger = logging.getLogger(__name__)


class ModelWorker:
    """Persistent per-model worker consuming jobs from a Redis list.

    Uses `BLPOP` on key `model:worker:{model_id}:queue` and expects JSON-serialized job dicts.
    """

    def __init__(self, model_id: str):
        self.model_id = model_id
        self._task: Optional[asyncio.Task] = None
        self._running = False

    def start(self) -> None:
        if not self._running:
            self._running = True
            self._task = asyncio.create_task(self._worker_loop())
            # Kick off a background preload of the model artifacts to surface mount/load errors early
            asyncio.create_task(self._preload_model())
            logger.info(f"[ModelWorker] Started worker for model {self.model_id}")

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    @property
    def _redis_key(self) -> str:
        return f"model:worker:{self.model_id}:queue"

    async def _worker_loop(self) -> None:
        websocket_manager = get_websocket_manager()
        redis = get_redis()
        while self._running:
            try:
                # BLPOP returns (key, value) when an item is available
                item = await redis.blpop(self._redis_key, timeout=5)
                if not item:
                    await asyncio.sleep(0.1)
                    continue

                _, raw = item
                try:
                    job = json.loads(raw)
                except Exception:
                    logger.exception("Failed to decode job payload, skipping")
                    continue

                task_id = job.get("task_id") or str(uuid.uuid4())
                user_id = job.get("user_id") or "anonymous"
                payload = job.get("payload", {})

                # Notify queued
                await websocket_manager.send_progress_update(task_id, {"status": "queued", "progress": 0}, user_id)

                try:
                    await websocket_manager.send_progress_update(task_id, {"status": "started", "progress": 5}, user_id)

                    # Notify building features / fetching data
                    await websocket_manager.send_progress_update(task_id, {"status": "building_features", "progress": 20, "message": "normalizing and building feature window"}, user_id)

                    # Normalize feature_window into a pandas.DataFrame if it's a raw list
                    fw = payload.get("feature_window")
                    if isinstance(fw, list):
                        try:
                            if len(fw) == 0:
                                df_fw = pd.DataFrame()
                            else:
                                first = fw[0]
                                if isinstance(first, (list, tuple)):
                                    # common row formats: [ts, o,h,l,c,v] or [o,h,l,c,v]
                                    # if first element looks like a timestamp (int/str) and length>=6, drop ts
                                    if len(first) >= 6 and (isinstance(first[0], (int, float, str))):
                                        row_vals = [list(r[1:]) if len(r) >= 6 else list(r) for r in fw]
                                    else:
                                        row_vals = [list(r) for r in fw]
                                    # try to infer columns by length
                                    ncols = len(row_vals[0])
                                    col_names = ["Open", "High", "Low", "Close", "Volume"][:ncols]
                                    df_fw = pd.DataFrame(row_vals, columns=col_names)
                                elif isinstance(first, dict):
                                    df_fw = pd.DataFrame(fw)
                                else:
                                    df_fw = pd.DataFrame(fw)

                            # Ensure no duplicate index
                            if df_fw.index.has_duplicates:
                                df_fw = df_fw.reset_index(drop=True)

                            # Coerce numeric columns where possible
                            for c in df_fw.columns:
                                try:
                                    df_fw[c] = pd.to_numeric(df_fw[c], errors="coerce")
                                except Exception:
                                    pass

                            payload["feature_window"] = df_fw
                        except Exception as norm_err:
                            logger.warning("Failed to normalize feature_window: %s", norm_err)

                    # Call proprietary runtime to perform inference
                    await websocket_manager.send_progress_update(task_id, {"status": "calculating_indices", "progress": 40, "message": "running TI and currency index calculations"}, user_id)

                    result = await proprietary_model_runtime.predict(
                        model_id=self.model_id,
                        symbol=payload.get("symbol"),
                        timeframe=payload.get("timeframe"),
                        feature_window=payload.get("feature_window"),
                        snr_features=payload.get("snr_features", {}),
                        supporting_ohlcv=payload.get("supporting_ohlcv", {}),
                    )

                    await websocket_manager.send_progress_update(task_id, {"status": "making_prediction", "progress": 75, "message": "running model forward pass"}, user_id)

                    # Completed
                    await websocket_manager.send_progress_update(task_id, {"status": "complete", "progress": 100, "result": result}, user_id)
                except ProprietaryModelContractError as pcm_err:
                    logger.warning(f"[ModelWorker:{self.model_id}] Contract error: {pcm_err}")
                    await websocket_manager.send_progress_update(task_id, {"status": "error", "progress": 0, "error": str(pcm_err), "missing_features": getattr(pcm_err, 'missing_features', None)}, user_id)
                except Exception as ex:
                    logger.exception(f"[ModelWorker:{self.model_id}] Inference failed: {ex}")
                    await websocket_manager.send_progress_update(task_id, {"status": "error", "progress": 0, "error": str(ex)}, user_id)

            except asyncio.CancelledError:
                break
            except Exception as loop_err:
                logger.exception(f"[ModelWorker:{self.model_id}] Worker loop error: {loop_err}")
                await asyncio.sleep(1)

    async def _preload_model(self) -> None:
        """Attempt to load the model artifact into the runtime caches on worker start.

        This helps surface mount or file-access errors early rather than per-job timeouts.
        """
        try:
            websocket_manager = get_websocket_manager()
            from app.core.ml.proprietary_model_runtime import proprietary_model_runtime

            # Resolve contract and artifact path; runs quickly in event loop
            try:
                contract = proprietary_model_runtime._contract(self.model_id)
                artifact_path = contract.get("artifact")
            except Exception as e:
                await websocket_manager.send_progress_update("preload:" + self.model_id, {"status": "preload_failed", "error": str(e)}, "system")
                logger.warning("[ModelWorker] Preload contract resolution failed for %s: %s", self.model_id, e)
                return

            # Emit debug info about the artifact path
            try:
                logger.info("[ModelWorker] Preloading model %s artifact=%s exists=%s", self.model_id, artifact_path, artifact_path.is_file() if artifact_path is not None else None)
            except Exception:
                logger.exception("[ModelWorker] Could not check artifact existence for %s", self.model_id)

            # Use to_thread to avoid blocking the event loop for heavy loads
            try:
                await websocket_manager.send_progress_update("preload:" + self.model_id, {"status": "preload_started"}, "system")
                await asyncio.to_thread(proprietary_model_runtime._load_model_cached, self.model_id, artifact_path)
                await websocket_manager.send_progress_update("preload:" + self.model_id, {"status": "preload_complete"}, "system")
                logger.info("[ModelWorker] Preloaded model %s", self.model_id)
            except Exception as load_err:
                logger.exception("[ModelWorker] Model preload failed for %s: %s", self.model_id, load_err)
                await websocket_manager.send_progress_update("preload:" + self.model_id, {"status": "preload_failed", "error": str(load_err)}, "system")
        except Exception:
            logger.exception("Unexpected error in _preload_model")


class ModelWorkerManager:
    """Manager that ensures a local consumer exists and provides enqueue via Redis."""

    _workers: Dict[str, ModelWorker] = {}

    @classmethod
    def get_worker(cls, model_id: str) -> ModelWorker:
        if model_id not in cls._workers:
            cls._workers[model_id] = ModelWorker(model_id)
            cls._workers[model_id].start()
        return cls._workers[model_id]

    @classmethod
    async def enqueue_job(cls, model_id: str, user_id: str, payload: Dict[str, Any]) -> str:
        redis = get_redis()
        task_id = str(uuid.uuid4())
        job = {"task_id": task_id, "user_id": user_id, "payload": payload}
        key = f"model:worker:{model_id}:queue"
        # Use RPUSH (append) and worker uses BLPOP to pop from left
        await redis.rpush(key, json.dumps(job))
        logger.info("[ModelWorkerManager] Enqueued job %s to %s (user=%s)", task_id, key, user_id)
        # Ensure a local worker exists to consume (for single-process deployment)
        cls.get_worker(model_id)
        return task_id

