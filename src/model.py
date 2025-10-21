import torch
import torch.nn as nn


# -----------------------------
# Encoder
# -----------------------------
class SpectrumEncoder(nn.Module):
    def __init__(self, input_dim=2000, embed_dim=256, n_layers=4, n_heads=4, ff_dim=1024):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads, dim_feedforward=ff_dim,
            dropout=0.1, batch_first=True, activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(self, x):
        # x: (batch, input_dim)
        x = self.input_proj(x).unsqueeze(1)   # (B, 1, D)
        x = self.encoder(x)                   # (B, 1, D)
        return x.squeeze(1)                   # (B, D)


# -----------------------------
# Counts-only model (mass derived from counts)
# -----------------------------
class Spectrum2Counts(nn.Module):
    def __init__(self, input_dim=2000, embed_dim=256,
                 n_layers=4, n_heads=4, ff_dim=1024,
                 num_elements=5, use_presence_head=True, use_presence_gating=True):
        """
        num_elements: number of element types (e.g., C,H,N,O,S)
        use_presence_head: optional binary presence head
        use_presence_gating: if True, multiply counts by presence mask
        """
        super().__init__()
        self.encoder = SpectrumEncoder(input_dim, embed_dim, n_layers, n_heads, ff_dim)

        # Element counts head (predict non‑negative counts via softplus)
        self.elem_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, num_elements),
            nn.Softplus()   # ensures non‑negative counts
        )

        # Optional presence head
        self.use_presence_head = use_presence_head
        self.use_presence_gating = use_presence_gating
        if use_presence_head:
            self.presence_head = nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, embed_dim // 2),
                nn.GELU(),
                nn.Linear(embed_dim // 2, num_elements)
            )
        else:
            self.presence_head = None

    def forward(self, spectrum):
        """
        Returns:
            elem_counts: (B, num_elements) -> predicted counts (>=0)
            presence_logits (optional): (B, num_elements)
        """
        memory = self.encoder(spectrum)            # (B, D)
        elem_counts = self.elem_head(memory)       # (B, E), already >=0

        if self.use_presence_head:
            presence_logits = self.presence_head(memory)  # (B, E)
            if self.use_presence_gating:
                presence_mask = torch.sigmoid(presence_logits)
                elem_counts = elem_counts * presence_mask
            return elem_counts, presence_logits

        return elem_counts