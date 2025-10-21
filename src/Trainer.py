import torch
import torch.nn as nn
import torch.nn.functional as F
from src.FormulaUtils import FormulaUtils


class Trainer:
    def __init__(self, model, optimizer, criterion, device, scaler,
                 lambda_elem_start=0.1, lambda_elem_max=0.5, lambda_mass=1e-4,
                 use_presence=True, ramp_epochs=10,
                 lambda_consistency=0.01, lambda_seq_mass=1e-4,
                 lambda_formula_rank=1e-3):
        """
        Trainer for Spectrum2Formula model.

        Args:
            model: Spectrum2FormulaHybrid
            optimizer: torch optimizer
            criterion: CE loss for sequence tokens
            device: torch.device
            scaler: GradScaler for mixed precision
            lambda_elem_start: starting weight for element loss
            lambda_elem_max: max weight for element loss
            lambda_mass: weight for auxiliary mass regression loss
            use_presence: whether to use presence head
            ramp_epochs: epochs to ramp element loss weight
            lambda_consistency: weight for consistency loss
            lambda_seq_mass: weight for sequence-level mass penalty
            lambda_formula_rank: weight for formula transformer ranking loss
        """
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.scaler = scaler

        # Loss weights
        self.lambda_elem_start = lambda_elem_start
        self.lambda_elem_max = lambda_elem_max
        self.lambda_mass = lambda_mass
        self.use_presence = use_presence
        self.ramp_epochs = ramp_epochs
        self.lambda_consistency = lambda_consistency
        self.lambda_seq_mass = lambda_seq_mass
        self.lambda_formula_rank = lambda_formula_rank

        if self.use_presence:
            self.presence_loss_fn = nn.BCEWithLogitsLoss()

    def _counts_from_tokens(self, tokens, tokenizer, element_order):
        decoded = tokenizer.decode(tokens.tolist())
        fdict = FormulaUtils.parse_formula_dict(decoded)
        return torch.tensor([fdict.get(e, 0) for e in element_order], dtype=torch.float32)

    def train_step(self, batch, elem_targets_all, mass_targets_all, batch_indices,
                   epoch=0, tokenizer=None, element_order=None,
                   candidate_sampler=None):
        """
        candidate_sampler: function that given a batch index returns a set of negative formulas
                           (as element count vectors). Should return tensor (K-1, num_elements).
        """
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        spectrum, tokens = batch
        spectrum, tokens = spectrum.to(self.device), tokens.to(self.device)
        tgt_in, tgt_out = tokens[:, :-1], tokens[:, 1:]

        # Targets
        elem_tgt = elem_targets_all[batch_indices].to(self.device)
        elem_tgt = torch.log1p(elem_tgt)
        mass_tgt = mass_targets_all[batch_indices].to(self.device) / 1000.0

        with torch.amp.autocast("cuda"):
            outputs = self.model(spectrum, tgt_in)
            if self.use_presence:
                logits, elem_logits, mass_pred, presence_logits = outputs
            else:
                logits, elem_logits, mass_pred = outputs
                presence_logits = None

            # --- Losses ---
            ce_loss = self.criterion(logits.reshape(-1, logits.size(-1)),
                                     tgt_out.reshape(-1))
            elem_loss = F.smooth_l1_loss(elem_logits, elem_tgt)
            mass_loss = F.mse_loss(mass_pred.squeeze(-1), mass_tgt)

            if self.use_presence:
                presence_tgt = (elem_tgt > 0).float()
                pres_loss = self.presence_loss_fn(presence_logits, presence_tgt)
            else:
                pres_loss = torch.tensor(0.0, device=self.device)

            # Consistency + sequence-level mass loss
            if tokenizer is not None and element_order is not None:
                counts_seq, seq_masses = [], []
                for i in range(tokens.size(0)):
                    decoded = tokenizer.decode(tokens[i].tolist())
                    fdict = FormulaUtils.parse_formula_dict(decoded)
                    counts_seq.append(torch.tensor([fdict.get(e, 0) for e in element_order],
                                                   dtype=torch.float32))
                    seq_masses.append(FormulaUtils.compute_mass(fdict) / 1000.0)
                counts_seq = torch.stack(counts_seq).to(self.device)
                counts_seq = torch.log1p(counts_seq)
                consistency_loss = F.mse_loss(elem_logits, counts_seq)

                seq_masses = torch.tensor(seq_masses, device=self.device, dtype=torch.float32)
                seq_mass_loss = F.mse_loss(seq_masses, mass_tgt)
            else:
                consistency_loss = torch.tensor(0.0, device=self.device)
                seq_mass_loss = torch.tensor(0.0, device=self.device)

            # --- Formula ranking loss (InfoNCE style) ---
            if self.model.use_formula_transformer and candidate_sampler is not None:
                B = spectrum.size(0)
                num_elements = elem_tgt.size(1)
                candidate_sets = []
                labels = []
                for i in range(B):
                    true_counts = elem_targets_all[batch_indices[i]].cpu()
                    negatives = candidate_sampler(batch_indices[i])  # (K-1, E)
                    candidates = torch.cat([true_counts.unsqueeze(0), negatives], dim=0)  # (K, E)
                    candidate_sets.append(candidates)
                    labels.append(0)  # true formula is at index 0
                candidate_counts = torch.stack(candidate_sets).to(self.device).float()
                labels = torch.tensor(labels, device=self.device, dtype=torch.long)

                energies = self.model.formula_head(self.model.encoder(spectrum),
                                                   torch.log1p(candidate_counts))
                rank_loss = F.cross_entropy(energies, labels)
            else:
                rank_loss = torch.tensor(0.0, device=self.device)

            # Ramp lambda_elem
            lambda_elem = min(self.lambda_elem_max,
                              self.lambda_elem_start + (epoch / self.ramp_epochs) *
                              (self.lambda_elem_max - self.lambda_elem_start))

            # Final loss
            loss = (ce_loss +
                    lambda_elem * elem_loss +
                    self.lambda_mass * mass_loss +
                    self.lambda_consistency * consistency_loss +
                    self.lambda_seq_mass * seq_mass_loss +
                    self.lambda_formula_rank * rank_loss +
                    (0.1 * pres_loss if self.use_presence else 0.0))

        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        return (ce_loss.item(),
                elem_loss.item(),
                mass_loss.item(),
                pres_loss.item() if self.use_presence else 0.0,
                consistency_loss.item(),
                seq_mass_loss.item(),
                rank_loss.item(),
                loss.item())