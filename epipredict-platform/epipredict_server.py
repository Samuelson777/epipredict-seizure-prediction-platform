"""
epipredict_server.py
FastAPI + WebSocket Streaming Server integrating BrainFlow Biosensor Acquisition
and the PyTorch EpiPredictHybridNet Engine.
Run with: uvicorn epipredict_server:app --host 0.0.0.0 --port 8000
"""

import asyncio
import time
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from epipredict_engine import (
    AdaptiveWaveletPreprocessor,
    EpiPredictConfig,
    EpiPredictHybridNet,
    MultimodalViterbiHMM,
    PersistenceRiskGate,
)

app = FastAPI(
    title="EpiPredict Neurological Predictive Analytics API",
    version="2.5.0",
    description="IEC 62304 Class C SaMD Real-Time EEG + ACC Telemetry & Seizure Prediction Server",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

cfg = EpiPredictConfig()
preproc = AdaptiveWaveletPreprocessor(cfg)
model = EpiPredictHybridNet(num_bands=6, time_bins=32)
model.eval()
hmm = MultimodalViterbiHMM(deadzone=cfg.acc_deadzone_threshold)
gate = PersistenceRiskGate(k=cfg.persistence_k, n=cfg.persistence_n, threshold=cfg.risk_threshold)
rng = np.random.default_rng(101)


@app.get("/api/health")
def health_check() -> Dict[str, object]:
    return {
        "status": "ONLINE",
        "samd_class": "IEC 62304 Class C",
        "sampling_rate_hz": cfg.sampling_rate,
        "sph_minutes": cfg.sph_minutes,
        "sop_minutes": cfg.sop_minutes,
    }


@app.websocket("/ws/telemetry")
async def telemetry_websocket(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        while True:
            t0 = time.perf_counter()
            eeg_win = rng.normal(0, 18.0, size=(cfg.num_channels, cfg.sampling_rate * cfg.window_seconds))
            acc_win = rng.normal(0, 0.25, size=(cfg.sampling_rate * cfg.window_seconds,))

            dwt_feat = preproc.compute_dynamic_dwt_features(eeg_win)
            rqa_det = preproc.compute_rqa_determinism(eeg_win[0])
            tensor_in = torch.from_numpy(dwt_feat).unsqueeze(0)

            with torch.no_grad():
                logits, ch_weights = model(tensor_in)
                prob = float(F.softmax(logits, dim=-1)[0, 1].item())

            alarm_triggered, k_active = gate.update(prob)
            hmm_state, _ = hmm.decode_viterbi(acc_win)
            top_ch = torch.topk(ch_weights[0], k=cfg.target_active_channels).indices.tolist()
            latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)

            payload = {
                "timestamp": time.time(),
                "preictal_prob": round(prob, 4),
                "rqa_determinism": round(rqa_det, 4),
                "hmm_state": int(hmm_state),
                "alarm_triggered": alarm_triggered,
                "persistence_count": k_active,
                "focal_channels": top_ch,
                "latency_ms": latency_ms,
            }
            await websocket.send_json(payload)
            await asyncio.sleep(1.0)
    except WebSocketDisconnect:
        pass