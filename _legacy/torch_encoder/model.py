import math
import torch

def sinusoidal_embed(x, d_model):
    """
    Sinusoidal embedding of continuous values (like normalized m/z).
    Args:
        x: (B, T) tensor of values in [0,1] or normalized range
        d_model: embedding dimension (must be even ideally)
    Returns:
        (B, T, d_model) sinusoidal embedding
    """
    device = x.device
    half = d_model // 2
    freqs = torch.exp(torch.linspace(0, math.log(10000), steps=half, device=device))
    x_expanded = x.unsqueeze(-1) * freqs  # (B, T, half)
    sin = torch.sin(x_expanded)
    cos = torch.cos(x_expanded)
    emb = torch.cat([sin, cos], dim=-1)
    if emb.size(-1) < d_model:
        pad = torch.zeros(x.shape[0], x.shape[1], d_model - emb.size(-1), device=device)
        emb = torch.cat([emb, pad], dim=-1)
    return emb

import torch
import torch.nn as nn

class AttentivePool(nn.Module):
    """
    Learned attention pooling over sequence of peaks.
    Produces a single spectrum embedding by attending over peaks.
    """
    def __init__(self, d_model):
        super().__init__()
        self.query = nn.Parameter(torch.randn(d_model))

    def forward(self, x, mask=None):
        """
        Args:
            x: (B, T, D) sequence of peak embeddings
            mask: (B, T) boolean mask, True for valid peaks
        Returns:
            (B, D) pooled embedding
        """
        scores = torch.einsum("btd,d->bt", x, self.query)  # (B, T)
        if mask is not None:
            scores = scores.masked_fill(~mask, -1e9)
        attn = torch.softmax(scores, dim=1)  # (B, T)
        pooled = torch.einsum("btd,bt->bd", x, attn)  # (B, D)
        return pooled

import torch
import torch.nn as nn
# from embeddings import sinusoidal_embed
# from pooling import AttentivePool

class MassSpectrumEncoder(nn.Module):
    """
    Transformer-based encoder over peak sequences to element count classification,
    with presence gating and m/z-informed embeddings.
    """
    def __init__(self,
                 d_model=256,
                 nhead=4,
                 num_layers=4,
                 dim_feedforward=1024,
                 max_peaks=500,
                 element_max_counts={'C': 50, 'H': 100, 'N': 20, 'O': 30, 'S': 10},
                 use_presence_head=True,
                 use_presence_gating=True,
                 dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.max_peaks = max_peaks
        self.element_max_counts = element_max_counts
        self.use_presence_head = use_presence_head
        self.use_presence_gating = use_presence_gating

        # Intensity projection
        self.intensity_proj = nn.Sequential(
            nn.Linear(2, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, d_model // 2),
        )
        # m/z projection
        self.mz_proj = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, d_model // 2),
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, activation="gelu"
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Attention pooling
        self.pool = AttentivePool(d_model)

        # Shared mixing
        self.shared_mlp = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # Element heads
        self.element_heads = nn.ModuleDict()
        for element, max_count in element_max_counts.items():
            self.element_heads[element] = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, max_count + 1)
            )

        # Presence head
        if self.use_presence_head:
            self.presence_head = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, len(element_max_counts))
            )
        else:
            self.presence_head = None

    def forward(self, mz_values, intensities, mask=None):
        B, T = mz_values.shape
        device = mz_values.device

        # Intensity features
        log_int = torch.log1p(intensities.clamp(min=0))
        clip_int = intensities.clamp(min=0, max=1.0)
        int_feat = torch.stack([log_int, clip_int], dim=-1)
        int_emb = self.intensity_proj(int_feat)

        # m/z sinusoidal embedding
        mz_emb_raw = sinusoidal_embed(mz_values, self.d_model)
        mz_emb = self.mz_proj(mz_emb_raw)

        # Peak token embedding
        x = torch.cat([mz_emb, int_emb], dim=-1)

        # Transformer encoding
        pad_mask = None
        if mask is not None:
            pad_mask = ~mask
        x = self.transformer(x, src_key_padding_mask=pad_mask)

        # Attention pooling
        pooled = self.pool(x, mask=mask)

        # Shared mixing
        shared = self.shared_mlp(pooled)

        # Presence head
        if self.use_presence_head:
            presence_logits = self.presence_head(shared)
            presence_prob = torch.sigmoid(presence_logits)
        else:
            presence_logits = None
            presence_prob = None

        # Element predictions
        predictions = {}
        for i, (element, head) in enumerate(self.element_heads.items()):
            logits = head(shared)
            if self.use_presence_head and self.use_presence_gating:
                nz_mask = torch.ones(logits.size(-1), device=device)
                nz_mask[0] = 0.0
                bias = (1.0 - presence_prob[:, i]).unsqueeze(-1) * nz_mask
                logits = logits - 5.0 * bias
            predictions[element] = logits

        if self.use_presence_head:
            return predictions, presence_logits
        return predictions