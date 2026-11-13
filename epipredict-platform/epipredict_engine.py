"""
epipredict_engine.py
Core Neuroengineering, Multi-Resolution Wavelet Pre-Processing, RQA Nonlinear Dynamics,
Hybrid Set-Transformer + DCNN-CBAM-BiLSTM Model, and Supervised Viterbi HMM Engine.
"""

import time
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pywt
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class EpiPredictConfig:
    sampling_rate: int = 256
    window_seconds: int = 5
    overlap_ratio: float = 0.50
    wavelet_name: str = "db4"
    num_channels: int = 18
    target_active_channels: int = 3
    sph_minutes: int = 15
    sop_minutes: int = 30
    persistence_k: int = 8
    persistence_n: int = 10
    risk_threshold: float = 0.78
    acc_deadzone_threshold: float = 0.08


class AdaptiveWaveletPreprocessor:
    """
    Executes Z-score normalization, dynamic multi-resolution DWT via PyWavelets,
    and Recurrence Quantification Analysis (RQA) determinism extraction.
    """

    def __init__(self, config: EpiPredictConfig):
        self.cfg = config
        self.wavelet = pywt.Wavelet(config.wavelet_name)

    def compute_dynamic_dwt_features(self, eeg_epoch: np.ndarray) -> np.ndarray:
        num_channels, num_samples = eeg_epoch.shape
        mean = np.mean(eeg_epoch, axis=-1, keepdims=True)
        std = np.std(eeg_epoch, axis=-1, keepdims=True) + 1e-8
        norm_eeg = (eeg_epoch - mean) / std

        max_level = pywt.dwt_max_level(data_len=num_samples, filter_len=self.wavelet.dec_len)
        decomp_level = min(max_level, 5)

        time_bins = 32
        feature_map = np.zeros((num_channels, decomp_level + 1, time_bins), dtype=np.float32)

        for c in range(num_channels):
            coeffs = pywt.wavedec(norm_eeg[c], self.wavelet, level=decomp_level)
            for b_idx, coeff in enumerate(coeffs):
                energy = np.square(coeff)
                x_old = np.linspace(0, 1, num=len(energy))
                x_new = np.linspace(0, 1, num=time_bins)
                feature_map[c, b_idx, :] = np.interp(x_new, x_old, energy)

        return feature_map

    @staticmethod
    def compute_rqa_determinism(signal_1d: np.ndarray, delay: int = 4, dim: int = 3, eps: float = 0.5) -> float:
        n = len(signal_1d)
        step = max(1, n // 128)
        sig = signal_1d[::step]
        sig = (sig - np.mean(sig)) / (np.std(sig) + 1e-8)
        m_len = len(sig) - (dim - 1) * delay
        if m_len <= 10:
            return 0.0
        traj = np.stack([sig[i * delay : i * delay + m_len] for i in range(dim)], axis=-1)
        dist = np.linalg.norm(traj[:, None, :] - traj[None, :, :], axis=-1)
        rec_mat = (dist < eps).astype(np.float32)
        np.fill_diagonal(rec_mat, 0.0)
        total_rec = np.sum(rec_mat)
        if total_rec == 0:
            return 0.0
        diag_rec = np.sum(rec_mat[:-1, :-1] * rec_mat[1:, 1:])
        return float(np.clip(diag_rec / total_rec, 0.0, 1.0))


class ClassBalancedFocalLoss(nn.Module):
    """Addresses severe CHB-MIT interictal vs. pre-ictal dataset imbalance."""

    def __init__(self, alpha: float = 0.85, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        alpha_t = torch.where(targets == 1, self.alpha, 1.0 - self.alpha)
        focal_loss = alpha_t * ((1.0 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


class ChannelAttentionModule(nn.Module):
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        mid = max(1, channels // reduction)
        self.mlp = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_pool = x.mean(dim=(-2, -1))
        max_pool = x.amax(dim=(-2, -1))
        attn = torch.sigmoid(self.mlp(avg_pool) + self.mlp(max_pool)).unsqueeze(-1).unsqueeze(-1)
        return x * attn


class SpatialSpectralAttentionModule(nn.Module):
    def __init__(self, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_out, max_out], dim=1)
        return x * torch.sigmoid(self.conv(concat))


class CBAMBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.ca = ChannelAttentionModule(channels)
        self.sa = SpatialSpectralAttentionModule()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sa(self.ca(x))


class EpiPredictHybridNet(nn.Module):
    """
    Two-Stage Channel-Aware Set Transformer + 2D-DCNN + CBAM + Bi-LSTM Architecture.
    """

    def __init__(self, num_bands: int = 6, time_bins: int = 32, hidden_dim: int = 64):
        super().__init__()
        in_dim = num_bands * time_bins
        self.channel_proj = nn.Linear(in_dim, hidden_dim)
        self.set_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, batch_first=True)

        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.cbam1 = CBAMBlock(16)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.cbam2 = CBAMBlock(32)

        self.pool = nn.AdaptiveAvgPool2d((4, 8))
        self.bilstm = nn.LSTM(
            input_size=32 * 4,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.2,
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, 2),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        b, c, h, w = x.shape
        flat_ch = x.view(b, c, h * w)
        ch_emb = self.channel_proj(flat_ch)
        _, attn_weights = self.set_attention(ch_emb, ch_emb, ch_emb)

        ch_scores = attn_weights.mean(dim=1).unsqueeze(-1).unsqueeze(-1)
        fused_scalogram = (x * ch_scores).sum(dim=1, keepdim=True)

        feat = self.cbam1(F.relu(self.bn1(self.conv1(fused_scalogram))))
        feat = self.cbam2(F.relu(self.bn2(self.conv2(feat))))
        feat = self.pool(feat)

        seq = feat.permute(0, 3, 1, 2).contiguous().view(b, 8, 32 * 4)
        lstm_out, _ = self.bilstm(seq)
        logits = self.classifier(lstm_out[:, -1, :])
        return logits, ch_scores.squeeze(-1).squeeze(-1)


class MultimodalViterbiHMM:
    """
    Supervised 3-State Hidden Markov Model fusing Triaxial Accelerometer (ACC)
    and Passive Infrared (PIR) sleep movement states via the Viterbi algorithm.
    """

    def __init__(self, deadzone: float = 0.08):
        self.deadzone = deadzone
        self.start_prob = np.array([0.85, 0.14, 0.01])
        self.trans_mat = np.array([
            [0.90, 0.09, 0.01],
            [0.65, 0.30, 0.05],
            [0.05, 0.10, 0.85],
        ])
        self.emit_mat = np.array([
            [0.92, 0.07, 0.01],
            [0.15, 0.75, 0.10],
            [0.01, 0.14, 0.85],
        ])

    def decode_viterbi(self, acc_window: np.ndarray) -> Tuple[int, List[int]]:
        thresholded = np.where(np.abs(acc_window) >= self.deadzone, acc_window, 0.0)
        sub_windows = np.array_split(thresholded, 5)
        obs_seq = []
        for sw in sub_windows:
            rms = float(np.sqrt(np.mean(np.square(sw))))
            obs_seq.append(0 if rms < 0.12 else (1 if rms < 0.45 else 2))

        num_states = 3
        T = len(obs_seq)
        viterbi = np.zeros((num_states, T))
        backpointer = np.zeros((num_states, T), dtype=int)

        viterbi[:, 0] = np.log(self.start_prob + 1e-12) + np.log(self.emit_mat[:, obs_seq[0]] + 1e-12)
        for t in range(1, T):
            for s in range(num_states):
                trans_p = viterbi[:, t - 1] + np.log(self.trans_mat[:, s] + 1e-12)
                best_prev = int(np.argmax(trans_p))
                viterbi[s, t] = trans_p[best_prev] + np.log(self.emit_mat[s, obs_seq[t]] + 1e-12)
                backpointer[s, t] = best_prev

        best_path = [0] * T
        best_path[-1] = int(np.argmax(viterbi[:, -1]))
        for t in range(T - 2, -1, -1):
            best_path[t] = int(backpointer[best_path[t + 1], t + 1])

        return best_path[-1], best_path


class PersistenceRiskGate:
    """Post-processing k-of-n persistence gate to minimize False Prediction Rate (FPR/h)."""

    def __init__(self, k: int = 8, n: int = 10, threshold: float = 0.78):
        self.k = k
        self.n = n
        self.threshold = threshold
        self.history = deque(maxlen=n)

    def update(self, prob: float) -> Tuple[bool, int]:
        self.history.append(1 if prob >= self.threshold else 0)
        exceed_count = sum(self.history)
        return exceed_count >= self.k, exceed_count


if __name__ == "__main__":
    cfg = EpiPredictConfig()
    preproc = AdaptiveWaveletPreprocessor(cfg)
    model = EpiPredictHybridNet(num_bands=6, time_bins=32)
    model.eval()
    hmm = MultimodalViterbiHMM(deadzone=cfg.acc_deadzone_threshold)
    gate = PersistenceRiskGate(k=cfg.persistence_k, n=cfg.persistence_n, threshold=cfg.risk_threshold)

    rng = np.random.default_rng(42)
    synthetic_eeg = rng.normal(0, 15.0, size=(cfg.num_channels, cfg.sampling_rate * cfg.window_seconds))
    synthetic_acc = rng.normal(0, 0.65, size=(cfg.sampling_rate * cfg.window_seconds,))

    t0 = time.perf_counter()
    dwt_tensor = preproc.compute_dynamic_dwt_features(synthetic_eeg)
    rqa_det = preproc.compute_rqa_determinism(synthetic_eeg[0])
    input_batch = torch.from_numpy(dwt_tensor).unsqueeze(0)

    with torch.no_grad():
        logits, ch_weights = model(input_batch)
        probs = F.softmax(logits, dim=-1)
        preictal_prob = float(probs[0, 1].item())

    alarm_active, k_count = gate.update(preictal_prob)
    motor_state, state_path = hmm.decode_viterbi(synthetic_acc)
    latency_s = time.perf_counter() - t0
    top_channels = torch.topk(ch_weights[0], k=cfg.target_active_channels).indices.tolist()

    print(f"[EpiPredict PyTorch Engine] Inference Complete in {latency_s:.4f} s")
    print(f" -> Pre-Ictal Probability   : {preictal_prob * 100:.2f}% (k-of-n Gate: {k_count}/{cfg.persistence_n})")
    print(f" -> RQA Determinism (DET)   : {rqa_det:.4f}")
    print(f" -> Focal Set-Trans Channels: {top_channels}")
    print(f" -> Viterbi HMM Motor State : S{motor_state} (Sequence: {state_path})")