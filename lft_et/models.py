"""Network definitions (state-dict compatible with the trained checkpoints) and the input encoder."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn


class Encoder:
    """Input standardisation and text vocabularies fitted on the training years (JSON export)."""

    def __init__(self, path: str | Path):
        s = json.loads(Path(path).read_text(encoding="utf-8"))
        self.ni = s["numeric_columns"]
        self.ci = s["categorical_columns"]
        self.mean = np.asarray(s["mean"], dtype="float64")
        self.scale = np.asarray(s["scale"], dtype="float64")
        self.vocab = {j: v for j, v in zip(self.ci, s["vocabulary"])}
        self.maps = [{t: i + 1 for i, t in enumerate(v)} for v in s["vocabulary"]]   # 0 = unknown text
        self.sizes = [len(v) + 1 for v in s["vocabulary"]]
        self.target_mean = s["target_mean"]
        self.target_scale = s["target_scale"]
        self.lags = s.get("lags", 0)

    def numeric(self, x: np.ndarray) -> np.ndarray:
        return ((x[:, :, self.ni] - self.mean) / self.scale).astype("float32")

    def ids(self, c: np.ndarray) -> np.ndarray:
        return np.stack([np.vectorize(m.get, otypes=[np.int64])(c[:, :, j], 0) for m, j in zip(self.maps, self.ci)], 2)

    def unknown_share(self, c: np.ndarray) -> float:
        """Share of weather/wind texts not seen in training (mapped to the unknown token)."""
        return float((self.ids(c)[:, :, :2] == 0).mean())


class LFT(nn.Module):
    """Lag-fusion Transformer: issue tokens of the current and earlier issues are pooled by masked attention per
    valid day, day-level context is added, and a Transformer encoder runs across the seven valid days."""

    def __init__(self, enc: Encoder, cfg: dict):
        super().__init__()
        w, d, J = cfg["width"], cfg["dropout"], enc.lags
        self.J = J
        pos = {col: i for i, col in enumerate(enc.ni)}
        self.cur = [pos[0], pos[1], pos[3]]
        self.lag = [[pos[10 + 4 * (j - 1) + v] for v in range(4)] for j in range(1, J + 1)]
        self.ctx = [pos[col] for col in (2, 4, 5, 6, 7, 8, 9)] + [i for i, col in enumerate(enc.ni) if col >= 10 + 4 * J]
        av = [10 + 4 * (j - 1) + 3 for j in range(1, J + 1)]
        self.register_buffer("av_m", torch.tensor([enc.mean[pos[col]] for col in av], dtype=torch.float32))
        self.register_buffer("av_s", torch.tensor([enc.scale[pos[col]] for col in av], dtype=torch.float32))
        base = {0: {v: i + 1 for i, v in enumerate(enc.vocab[0])}, 1: {v: i + 1 for i, v in enumerate(enc.vocab[1])}}
        for k, j in enumerate(enc.ci):   # earlier issues share the current-issue text embeddings
            kind = 0 if j == 0 or (j >= 3 and (j - 3) % 2 == 0) else 1 if j != 2 else None
            if kind is not None:
                self.register_buffer(f"map{k}", torch.tensor([0] + [base[kind].get(v, 0) for v in enc.vocab[j]], dtype=torch.long))
        self.n_cat = len(enc.ci)
        sizes = enc.sizes
        self.wemb, self.vemb, self.zemb = nn.Embedding(sizes[0], w), nn.Embedding(sizes[1], w), nn.Embedding(sizes[2], w)
        self.tok = nn.Linear(6, w)
        self.age = nn.Parameter(torch.zeros(J + 1, w))
        self.q = nn.Parameter(torch.zeros(1, 1, w))
        self.pool = nn.MultiheadAttention(w, cfg["heads"], dropout=d, batch_first=True)
        self.n0 = nn.LayerNorm(w)
        self.ctxp = nn.Linear(len(self.ctx), w)
        self.pos = nn.Parameter(torch.zeros(1, 7, w))
        self.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(w, cfg["heads"], 4 * w, d, batch_first=True, norm_first=True), cfg["layers"], enable_nested_tensor=False)
        self.nq = len(cfg.get("quantiles", []) or [0])
        self.head = nn.Sequential(nn.LayerNorm(w), nn.Linear(w, self.nq))

    def forward(self, num: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
        B = num.shape[0]
        I = [getattr(self, f"map{k}")[ids[:, :, k]] if hasattr(self, f"map{k}") else ids[:, :, k] for k in range(self.n_cat)]
        cur = num[:, :, self.cur]
        toks = [torch.cat([cur, torch.zeros_like(cur[..., :2]), torch.ones_like(cur[..., :1])], -1)]
        mask = [torch.zeros(B, 7, dtype=torch.bool, device=num.device)]
        for j, c in enumerate(self.lag):
            v = num[:, :, c[:3]]
            a = (num[:, :, c[3]] * self.av_s[j] + self.av_m[j]) > 0.5
            toks.append(torch.cat([v, v[..., :2] - cur[..., :2], a.unsqueeze(-1).float()], -1))
            mask.append(~a)
        T = torch.stack(toks, 2)
        wv = torch.stack([self.wemb(I[0]) + self.vemb(I[1])] + [self.wemb(I[3 + 2 * j]) + self.vemb(I[4 + 2 * j]) for j in range(self.J)], 2)
        z = (self.tok(T) + wv + self.age).flatten(0, 1)
        m = torch.stack(mask, 2).flatten(0, 1)
        pooled = self.pool(self.q.expand(z.shape[0], 1, -1) + z[:, :1], z, z, key_padding_mask=m, need_weights=False)[0][:, 0]
        h = self.n0(pooled + z[:, 0]).view(B, 7, -1) + self.ctxp(num[:, :, self.ctx]) + self.zemb(I[2]).mean(1, keepdim=True) + self.pos
        o = self.head(self.enc(h))
        return o.squeeze(-1) if self.nq == 1 else o


class GRU(nn.Module):
    """Recurrent baseline on the current issue only (numerics + one-hot texts)."""

    def __init__(self, enc: Encoder, cfg: dict):
        super().__init__()
        self.sizes = enc.sizes
        features = len(enc.ni) + sum(enc.sizes)
        w, n, d = cfg["width"], cfg["layers"], cfg["dropout"]
        self.net = nn.GRU(features, w, num_layers=n, dropout=d if n > 1 else 0, batch_first=True)
        self.head = nn.Sequential(nn.Dropout(d), nn.Linear(w, 7))

    def forward(self, num: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
        x = torch.cat([num] + [nn.functional.one_hot(ids[:, :, k], n).float() for k, n in enumerate(self.sizes)], 2)
        return self.head(self.net(x)[1][-1])


def load_family(model_dir: str | Path, family: str):
    """Load encoder and all seed checkpoints of a model family ('lft', 'gru' or 'lftq')."""
    model_dir = Path(model_dir)
    enc = Encoder(model_dir / f"{family}_encoder.json")
    meta = json.loads((model_dir / f"{family}_config.json").read_text())
    nets = []
    for seed in meta["seeds"]:
        net = (GRU if family == "gru" else LFT)(enc, meta["config"])
        state = torch.load(model_dir / f"{family}_s{seed}.pt", map_location="cpu")
        if family == "gru":
            state = {k: v for k, v in state.items()}
        net.load_state_dict(state)
        net.eval()
        nets.append(net)
    return enc, meta, nets
