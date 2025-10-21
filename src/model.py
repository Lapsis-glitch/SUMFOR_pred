import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# Encoder
# -----------------------------
class SpectrumEncoder(nn.Module):
    def __init__(self, input_dim=2000, embed_dim=512, n_layers=6, n_heads=8, ff_dim=2048):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads, dim_feedforward=ff_dim,
            dropout=0.1, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(self, x):
        # x: (batch, input_dim)
        x = self.input_proj(x).unsqueeze(1)   # (batch, seq=1, embed_dim)
        x = self.encoder(x)                   # (batch, seq, embed_dim)
        return x.squeeze(1)                   # (batch, embed_dim)


# -----------------------------
# Decoder
# -----------------------------
class FormulaDecoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=512, n_layers=6, n_heads=8, ff_dim=2048):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim, nhead=n_heads, dim_feedforward=ff_dim,
            dropout=0.1, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
        self.fc_out = nn.Linear(embed_dim, vocab_size)

    def forward(self, tgt, memory):
        # tgt: (batch, seq_len), memory: (batch, embed_dim)
        tgt_emb = self.embedding(tgt)                    # (batch, seq_len, embed_dim)
        out = self.decoder(tgt_emb, memory.unsqueeze(1)) # memory: (batch, 1, embed_dim)
        return self.fc_out(out)                          # (batch, seq_len, vocab_size)


# -----------------------------
# Formula Transformer Head (energy-based scoring)
# -----------------------------
class FormulaTransformerHead(nn.Module):
    """
    Scores candidate formulas against the spectrum embedding.

    Inputs:
      - spectrum_memory: (B, D) from encoder
      - candidate_counts: (B, K, E) element counts per candidate (recommend log1p scale)
      - adduct_idx: optional (B, K) indices for adduct embedding

    Output:
      - energies: (B, K) higher = better match
    """
    def __init__(self, num_elements=6, d_model=512, formula_emb_dim=512,
                 use_adducts=False, num_adducts=0):
        super().__init__()
        self.num_elements = num_elements
        self.use_adducts = use_adducts

        self.formula_proj = nn.Sequential(
            nn.LayerNorm(num_elements),
            nn.Linear(num_elements, formula_emb_dim),
            nn.GELU(),
            nn.Linear(formula_emb_dim, d_model)
        )

        if use_adducts and num_adducts > 0:
            self.adduct_emb = nn.Embedding(num_adducts, d_model)
        else:
            self.adduct_emb = None

        # scoring over [spec_emb || formula_emb || interaction]
        self.scorer = nn.Sequential(
            nn.LayerNorm(3 * d_model),
            nn.Linear(3 * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1)
        )

    def forward(self, spectrum_memory, candidate_counts, adduct_idx=None):
        B, K, E = candidate_counts.size()
        spec = spectrum_memory                         # (B, D)
        form_emb = self.formula_proj(candidate_counts) # (B, K, D)

        if self.adduct_emb is not None and adduct_idx is not None:
            add_emb = self.adduct_emb(adduct_idx)      # (B, K, D)
            form_emb = form_emb + add_emb

        spec_expanded = spec.unsqueeze(1).expand(B, K, spec.size(-1))   # (B, K, D)
        interaction = spec_expanded * form_emb                          # (B, K, D)
        concat = torch.cat([spec_expanded, form_emb, interaction], dim=-1)  # (B, K, 3D)

        energy = self.scorer(concat).squeeze(-1)  # (B, K)
        return energy


# -----------------------------
# Full Model with Auxiliary Heads and Formula Scoring
# -----------------------------
class Spectrum2Formula(nn.Module):
    def __init__(self, input_dim, vocab_size, embed_dim=512,
                 n_layers=6, n_heads=8, ff_dim=2048,
                 num_elements=6, use_presence_head=True, hint_tokens=5,
                 use_formula_transformer=True, formula_emb_dim=512,
                 use_adducts=False, num_adducts=0):
        """
        num_elements: number of element types (e.g., C,H,N,O,S,P)
        use_presence_head: add a binary presence head for curriculum
        hint_tokens: number of early decoder tokens to average for decoder hint
        use_formula_transformer: enable energy-based formula scoring head
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_elements = num_elements
        self.use_presence_head = use_presence_head
        self.hint_tokens = hint_tokens
        self.use_formula_transformer = use_formula_transformer

        # Encoder & Decoder
        self.encoder = SpectrumEncoder(input_dim=input_dim, embed_dim=embed_dim,
                                       n_layers=n_layers, n_heads=n_heads, ff_dim=ff_dim)
        self.decoder = FormulaDecoder(vocab_size, embed_dim=embed_dim,
                                      n_layers=n_layers, n_heads=n_heads, ff_dim=ff_dim)

        # Gated fusion between encoder memory and a projected decoder hint
        self.hint_proj = nn.Linear(embed_dim, embed_dim)
        self.gate = nn.Sequential(
            nn.Linear(2 * embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 1),
            nn.Sigmoid()
        )

        # Element counts head (predict log1p(counts))
        self.elem_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, num_elements)
        )

        # Optional presence head (binary logits)
        if self.use_presence_head:
            self.presence_head = nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, embed_dim // 2),
                nn.GELU(),
                nn.Linear(embed_dim // 2, num_elements)
            )
        else:
            self.presence_head = None

        # Precursor mass regression
        self.mass_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, 1)
        )

        # Formula Transformer Head (energy-based scoring)
        if self.use_formula_transformer:
            self.formula_head = FormulaTransformerHead(
                num_elements=num_elements, d_model=embed_dim, formula_emb_dim=formula_emb_dim,
                use_adducts=use_adducts, num_adducts=num_adducts
            )
        else:
            self.formula_head = None

    def forward(self, spectrum, tgt):
        """
        Returns:
            logits: (batch, seq_len, vocab_size)
            elem_logits: (batch, num_elements)  -> interpret as log1p(counts)
            mass_pred: (batch, 1)
            presence_logits (optional): (batch, num_elements) -> raw logits
        """
        # Encoder memory from spectrum
        memory = self.encoder(spectrum)  # (batch, embed_dim)

        # Decode tokens (for CE)
        logits = self.decoder(tgt, memory)  # (batch, seq_len, vocab_size)

        # Build a decoder hint from early tokens (teacher-forced)
        k = min(self.hint_tokens, tgt.size(1))
        dec_hint = self.decoder.embedding(tgt[:, :k]).mean(dim=1)  # (batch, embed_dim)
        dec_hint_proj = self.hint_proj(dec_hint)

        # Gate decides how much to trust memory vs decoder hint
        gate_in = torch.cat([memory, dec_hint_proj], dim=-1)  # (batch, 2*embed_dim)
        alpha = self.gate(gate_in)                            # (batch, 1), in [0,1]
        fused = alpha * memory + (1 - alpha) * dec_hint_proj  # (batch, embed_dim)

        # Element head predicts log1p(counts)
        elem_logits = self.elem_head(fused)  # (batch, num_elements)

        # Mass regression from memory
        mass_pred = self.mass_head(memory)   # (batch, 1)

        if self.use_presence_head:
            presence_logits = self.presence_head(fused)  # (batch, num_elements)
            return logits, elem_logits, mass_pred, presence_logits

        return logits, elem_logits, mass_pred

    # Expose parts for beam search (backward compatible with your decoder)
    def encoder_forward(self, spectrum):
        return self.encoder(spectrum)  # (batch, embed_dim)

    def decoder_forward(self, tgt, memory):
        return self.decoder(tgt, memory)  # (batch, seq_len, vocab_size)

    # Formula scoring head: energy-based ranking of candidate formulas
    @torch.no_grad()
    def score_formulas(self, spectrum, candidate_counts, adduct_idx=None):
        """
        spectrum: (B, input_dim)
        candidate_counts: (B, K, num_elements), recommend log1p(counts)
        adduct_idx: optional (B, K) long tensor indices
        Returns: energies (B, K) higher = better
        """
        memory = self.encoder(spectrum)  # (B, D)
        if self.formula_head is None:
            raise RuntimeError("FormulaTransformerHead not initialized; set use_formula_transformer=True.")
        return self.formula_head(memory, candidate_counts, adduct_idx)