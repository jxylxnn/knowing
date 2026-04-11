"""
Transformer Model for NBA Player Stats Prediction.

Uses self-attention mechanism to capture long-term dependencies in
player performance trajectories. Supports Flash Attention for
significant speedup on Ampere+ GPUs.

Optimizations included:
- Flash Attention / scaled_dot_product_attention for faster inference
- Gradient checkpointing for memory efficiency
- BF16/FP16 mixed precision training
- torch.compile support for PyTorch 2.0+
- TF32 acceleration on Ampere+ GPUs
- Optimal DataLoader worker configuration
"""

import logging
import math
from contextlib import nullcontext
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader, Dataset

from src.models.gpu_utils import (
    get_device,
    WarmupCosineScheduler,
    apply_compile,
    get_compile_status,
    get_optimal_dataloader_workers,
    is_bf16_supported,
    get_autocast_dtype,
    create_grad_scaler,
    check_flash_attention_available,
)

logger = logging.getLogger(__name__)

# Enable cuDNN benchmark for optimal convolution algorithms
cudnn.benchmark = True

# Check if Flash Attention is available
_FLASH_ATTENTION_AVAILABLE = check_flash_attention_available()


def _safe_sdpa_context():
    """Return a context that prefers math SDPA kernels on CUDA when possible."""
    if not torch.cuda.is_available():
        return nullcontext()

    backend_cuda = getattr(torch.backends, "cuda", None)
    if backend_cuda is not None:
        sdp_kernel = getattr(backend_cuda, "sdp_kernel", None)
        if callable(sdp_kernel):
            try:
                return sdp_kernel(
                    enable_flash=False, enable_mem_efficient=False, enable_math=True
                )
            except TypeError:
                try:
                    return sdp_kernel(False, False, True)
                except Exception:
                    return nullcontext()
            except Exception:
                return nullcontext()

    attention_module = getattr(torch.nn, "attention", None)
    if attention_module is not None:
        sdpa_kernel = getattr(attention_module, "sdpa_kernel", None)
        sdp_backend = getattr(attention_module, "SDPBackend", None)
        if (
            callable(sdpa_kernel)
            and sdp_backend is not None
            and hasattr(sdp_backend, "MATH")
        ):
            try:
                return sdpa_kernel(sdp_backend.MATH)
            except Exception:
                return nullcontext()

    return nullcontext()


class PlayerSequenceDataset(Dataset):
    """Dataset for sequence-based player stat prediction."""

    def __init__(self, sequences: np.ndarray, targets: np.ndarray):
        self.sequences = torch.from_numpy(np.asarray(sequences, dtype=np.float32))
        self.targets = torch.from_numpy(np.asarray(targets, dtype=np.float32))

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int):
        return self.sequences[idx], self.targets[idx]


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].size(1)])
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class TransformerModel(nn.Module):
    """Transformer for processing long player stat trajectories with Self-Attention."""

    def __init__(
        self,
        input_dim: int,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 4,
        output_dim: int = 6,
        dropout: float = 0.1,
        dim_feedforward: int = 512,
        grad_checkpoint: bool = False,
    ):
        super(TransformerModel, self).__init__()

        self.d_model = d_model
        self.grad_checkpoint = grad_checkpoint

        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model,
            nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        self.fc = nn.Linear(d_model, output_dim)
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Xavier uniform."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, return_attention=False):
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)

        if self.grad_checkpoint and self.training:
            x = checkpoint(self.transformer_encoder, x, use_reentrant=False)
        else:
            x = self.transformer_encoder(x)

        out = self.fc(x[:, -1, :])
        return out


class TransformerWrapper:
    """Wrapper for handling sequence processing and training for the Transformer."""

    DEFAULT_CONFIG = {
        "d_model": 128,
        "nhead": 8,
        "num_layers": 4,
        "dim_feedforward": 512,
        "dropout": 0.1,
        "batch_size": 256,
        "epochs": 50,
        "lr": 5e-4,
        "warmup_ratio": 0.1,
        "grad_checkpoint": False,
        "use_compile": False,
        "allow_compile": False,
        "validation_use_eager": True,
        "validation_force_safe_sdpa": True,
    }

    def __init__(
        self,
        input_dim: int,
        seq_len: int = 50,
        config: Optional[Dict[str, Any]] = None,
        output_dim: Optional[int] = None,
    ):
        self.seq_len = seq_len
        self.device = get_device()

        self.config = {**self.DEFAULT_CONFIG, **(config or {})}
        self.output_dim = int(
            output_dim if output_dim is not None else self.config.get("output_dim", 6)
        )
        self.config["output_dim"] = self.output_dim

        self.eager_model = TransformerModel(
            input_dim=input_dim,
            d_model=self.config["d_model"],
            nhead=self.config["nhead"],
            num_layers=self.config["num_layers"],
            output_dim=self.output_dim,
            dim_feedforward=self.config["dim_feedforward"],
            dropout=self.config["dropout"],
            grad_checkpoint=self.config["grad_checkpoint"],
        ).to(self.device)

        self.model = self.eager_model
        self.validation_model = self.eager_model
        self.compile_enabled = False

        # Keep torch.compile opt-in and disabled unless the caller explicitly
        # allows it on the current environment.
        requested_compile = bool(self.config.get("use_compile", False))
        allow_compile = bool(self.config.get("allow_compile", False))
        if requested_compile and allow_compile and get_compile_status():
            compiled_model = apply_compile(self.eager_model, True, "Transformer")
            if compiled_model is not self.eager_model:
                self.model = compiled_model
                self.compile_enabled = True
        elif requested_compile and not allow_compile:
            logger.info(
                "Transformer torch.compile disabled by safety flag; using eager model."
            )

        self.input_dim = input_dim
        self.is_trained = False
        self.feat_mean = None
        self.feat_std = None
        self.validation_use_eager = bool(self.config.get("validation_use_eager", True))
        self.validation_force_safe_sdpa = bool(
            self.config.get("validation_force_safe_sdpa", True)
        )

        logger.info(
            f"Transformer initialized: d_model={self.config['d_model']}, "
            f"heads={self.config['nhead']}, layers={self.config['num_layers']}, "
            f"ff_dim={self.config['dim_feedforward']}, grad_ckpt={self.config['grad_checkpoint']}, "
            f"compile={self.compile_enabled}, validation_eager={self.validation_use_eager}, "
            f"safe_sdpa={self.validation_force_safe_sdpa}, device={self.device}"
        )

    def _create_sequences(
        self, df: pd.DataFrame, feature_cols: List[str], target_cols: List[str]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Creates sliding window sequences for each player (vectorized)."""
        df_sorted = df.sort_values(["PLAYER_ID", "GAME_DATE"])
        sequences = []
        targets = []

        for _, group in df_sorted.groupby("PLAYER_ID", sort=False):
            if len(group) < self.seq_len + 1:
                continue

            features = (
                group[feature_cols]
                .apply(pd.to_numeric, errors="coerce")
                .fillna(0)
                .values.astype(np.float32)
            )
            targets_raw = group[target_cols].apply(pd.to_numeric, errors="coerce")
            valid_target_rows = targets_raw.notna().all(axis=1).values
            targets_arr = targets_raw.fillna(0).values.astype(np.float32)

            for idx in range(self.seq_len, len(group)):
                if not valid_target_rows[idx]:
                    continue
                sequences.append(features[idx - self.seq_len : idx])
                targets.append(targets_arr[idx])

        if not sequences:
            return np.array([]), np.array([])

        return np.asarray(sequences, dtype=np.float32), np.asarray(
            targets, dtype=np.float32
        )

    def fit(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        target_cols: List[str],
        epochs: Optional[int] = None,
    ):
        """Train with warmup + cosine decay and gradient clipping."""
        epochs = epochs or self.config["epochs"]
        batch_size = self.config["batch_size"]
        lr = self.config["lr"]
        warmup_ratio = self.config["warmup_ratio"]

        logger.info(f"Generating sequences (len={self.seq_len}) for Transformer...")
        X, y = self._create_sequences(df, feature_cols, target_cols)

        if len(X) == 0:
            logger.warning("Not enough data to create sequences for Transformer.")
            return

        self.feat_mean = X.mean(axis=(0, 1))
        self.feat_std = X.std(axis=(0, 1)) + 1e-6
        X = (X - self.feat_mean) / self.feat_std

        dataset = PlayerSequenceDataset(X, y)
        pin_memory = self.device.type == "cuda"
        num_workers = get_optimal_dataloader_workers() if pin_memory else 0
        loader_kwargs = {
            "batch_size": batch_size,
            "shuffle": True,
            "pin_memory": pin_memory,
            "num_workers": num_workers,
            "drop_last": True,
        }
        if num_workers > 0:
            loader_kwargs["persistent_workers"] = True
            loader_kwargs["prefetch_factor"] = 2
        loader = DataLoader(dataset, **loader_kwargs)

        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        total_steps = epochs * len(loader)
        warmup_steps = int(total_steps * warmup_ratio)
        scheduler = WarmupCosineScheduler(optimizer, warmup_steps, total_steps)
        criterion = nn.MSELoss()

        device_str = self.device.type
        use_amp = device_str == "cuda"
        amp_dtype = get_autocast_dtype() if use_amp else None
        grad_scaler = create_grad_scaler(device_str, enabled=use_amp)

        best_loss = float("inf")
        patience_counter = 0
        early_stop_patience = 10

        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch_X, batch_y in loader:
                batch_X = batch_X.to(self.device, non_blocking=True)
                batch_y = batch_y.to(self.device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast(
                    device_type=device_str, dtype=amp_dtype, enabled=use_amp
                ):
                    preds = self.model(batch_X)
                    loss = criterion(preds, batch_y)
                if grad_scaler is not None:
                    grad_scaler.scale(loss).backward()
                    grad_scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_norm=1.0
                    )
                    grad_scaler.step(optimizer)
                    grad_scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_norm=1.0
                    )
                    optimizer.step()
                scheduler.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(loader)
            if avg_loss < best_loss - 1e-4:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if (epoch + 1) % 10 == 0:
                current_lr = optimizer.param_groups[0]["lr"]
                logger.info(
                    f"Transformer Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}, LR: {current_lr:.2e}"
                )

            if patience_counter >= early_stop_patience:
                logger.info(
                    f"Transformer early stopping at epoch {epoch + 1} (best loss: {best_loss:.4f})"
                )
                break

        self.is_trained = True
        logger.info(f"Transformer training complete. Final loss: {best_loss:.4f}")

    def _predict_sequences(
        self, sequences: np.ndarray, *, use_eager: bool = True
    ) -> np.ndarray:
        if not self.is_trained:
            return None

        sequences = np.asarray(sequences, dtype=np.float32)
        if sequences.size == 0:
            return np.empty((0, self.output_dim), dtype=np.float32)

        if sequences.ndim == 2:
            sequences = np.expand_dims(sequences, axis=0)

        model = (
            self.eager_model if use_eager or not self.compile_enabled else self.model
        )
        model.eval()

        sequences = (sequences - self.feat_mean) / self.feat_std
        seq_tensor = torch.from_numpy(sequences.astype(np.float32)).to(self.device)

        device_str = self.device.type
        use_amp = device_str == "cuda"
        amp_dtype = get_autocast_dtype() if use_amp else None
        sdpa_context = (
            _safe_sdpa_context()
            if (use_eager and self.validation_force_safe_sdpa and use_amp)
            else nullcontext()
        )

        with (
            torch.no_grad(),
            sdpa_context,
            torch.amp.autocast(
                device_type=device_str, dtype=amp_dtype, enabled=use_amp
            ),
        ):
            preds = model(seq_tensor).detach().cpu().numpy()
        return preds

    def predict(self, sequence: np.ndarray) -> np.ndarray:
        if not self.is_trained:
            return None
        preds = self._predict_sequences(sequence, use_eager=self.validation_use_eager)
        if preds is None:
            return None
        return preds

    def predict_batch(
        self, sequences: np.ndarray, *, use_eager: Optional[bool] = None
    ) -> np.ndarray:
        """Predict a batch of sequences, defaulting to the eager validation path."""
        if use_eager is None:
            use_eager = self.validation_use_eager
        return self._predict_sequences(sequences, use_eager=use_eager)

    def save(self, path: str):
        import joblib

        state = {
            "model_state": self.eager_model.state_dict(),
            "feat_mean": self.feat_mean,
            "feat_std": self.feat_std,
            "input_dim": self.input_dim,
            "seq_len": self.seq_len,
            "output_dim": self.output_dim,
            "config": self.config,
        }
        joblib.dump(state, path)
        logger.info(f"Transformer model saved to {path}")

    @classmethod
    def load(cls, path: str):
        import joblib

        state = joblib.load(path)

        config = state.get("config", {})
        model_state = state.get("model_state", {})

        if not config and model_state:
            config = cls._infer_config_from_state(model_state)

        output_dim = state.get("output_dim")
        if output_dim is None:
            fc_weight = model_state.get("fc.weight")
            if fc_weight is not None:
                output_dim = int(fc_weight.shape[0])
            else:
                output_dim = int(config.get("output_dim", 6))
        instance = cls(
            input_dim=state["input_dim"],
            seq_len=state["seq_len"],
            config=config,
            output_dim=output_dim,
        )
        instance.model.load_state_dict(model_state)
        if hasattr(instance, "eager_model"):
            instance.eager_model.load_state_dict(model_state)
        instance.feat_mean = state["feat_mean"]
        instance.feat_std = state["feat_std"]
        instance.is_trained = True
        return instance

    @staticmethod
    def _infer_config_from_state(model_state: Dict) -> Dict:
        d_model = None
        num_layers = 0
        nhead = None
        dim_feedforward = None
        for key, tensor in model_state.items():
            if key == "embedding.weight":
                d_model = int(tensor.shape[0])
            if key.startswith("transformer_encoder.layers."):
                parts = key.split(".")
                if len(parts) >= 3 and parts[2].isdigit():
                    layer_idx = int(parts[2])
                    if layer_idx > num_layers:
                        num_layers = layer_idx
                if ".linear1.weight" in key and dim_feedforward is None:
                    dim_feedforward = int(tensor.shape[0])
                if ".self_attn.in_proj_weight" in key and nhead is None and d_model:
                    nhead = 4
                    for candidate in [4, 8, 2, 1, d_model]:
                        if d_model % candidate == 0:
                            nhead = candidate
                            break
        num_layers += 1
        inferred = {}
        if d_model is not None:
            inferred["d_model"] = d_model
        if nhead is not None:
            inferred["nhead"] = nhead
        if num_layers > 0:
            inferred["num_layers"] = num_layers
        if dim_feedforward is not None:
            inferred["dim_feedforward"] = dim_feedforward
        return inferred
