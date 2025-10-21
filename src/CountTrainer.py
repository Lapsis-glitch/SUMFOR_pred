# src/CountTrainer.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.FormulaUtils import FormulaUtils


class CountTrainer:
    def __init__(self, model, optimizer, device, scaler,
                 lambda_elem_start=0.2, lambda_elem_max=0.8,
                 lambda_mass_penalty=0.2,
                 lambda_presence=0.1,
                 lambda_diversity=0.02,
                 lambda_entropy=0.01,
                 use_presence=True, ramp_epochs=10,
                 element_order=None, grad_clip=1.0,
                 element_weights=None):
        """
        Trainer for Spectrum2Counts (Poisson counts; mass from counts only).
        """
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.scaler = scaler

        self.lambda_elem_start = lambda_elem_start
        self.lambda_elem_max = lambda_elem_max
        self.lambda_mass_penalty = lambda_mass_penalty
        self.lambda_presence = lambda_presence
        self.lambda_diversity = lambda_diversity
        self.lambda_entropy = lambda_entropy

        self.use_presence = use_presence
        self.ramp_epochs = ramp_epochs
        self.element_order = element_order
        self.grad_clip = grad_clip

        self.poisson_nll = nn.PoissonNLLLoss(log_input=False, reduction="none")
        if element_weights is not None:
            self.element_weights = torch.tensor(element_weights, dtype=torch.float32, device=device)
        else:
            self.element_weights = None

        if self.use_presence:
            self.presence_loss_fn = nn.BCEWithLogitsLoss()

    def _batch_diversity_penalty(self, elem_rate):
        # Encourage per-element std across batch
        if elem_rate.size(0) <= 1:
            return torch.tensor(0.0, device=elem_rate.device)
        std = elem_rate.std(dim=0)
        return torch.mean(1.0 / (std + 1e-3))

    def _presence_entropy(self, presence_logits):
        # Encourage non-degenerate presence probabilities
        p = torch.sigmoid(presence_logits).clamp(1e-6, 1 - 1e-6)
        ent = -(p * torch.log(p) + (1 - p) * torch.log(1 - p))
        return ent.mean()

    def train_step(self, batch, elem_targets_all, mass_targets_all, batch_indices, epoch=0):
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        spectrum, _counts = batch
        spectrum = spectrum.to(self.device)

        elem_tgt = elem_targets_all[batch_indices].to(self.device).float()
        mass_tgt = mass_targets_all[batch_indices].to(self.device) / 1000.0  # kDa

        with torch.amp.autocast("cuda"):
            outputs = self.model(spectrum)
            if self.use_presence:
                elem_rate, presence_logits = outputs  # elem_rate >= 0 (Softplus)
            else:
                elem_rate = outputs
                presence_logits = None

            # Poisson NLL on counts
            elem_loss_per = self.poisson_nll(elem_rate + 1e-6, elem_tgt)  # (B, E)
            if self.element_weights is not None:
                elem_loss = (elem_loss_per * self.element_weights).mean()
            else:
                elem_loss = elem_loss_per.mean()

            # Presence losses
            if self.use_presence:
                presence_tgt = (elem_tgt > 0).float()
                pres_loss = self.presence_loss_fn(presence_logits, presence_tgt)
                ent_loss = self._presence_entropy(presence_logits)
            else:
                pres_loss = torch.tensor(0.0, device=self.device)
                ent_loss = torch.tensor(0.0, device=self.device)

            # Mass penalty (relative error) from counts
            pred_mass = FormulaUtils.compute_mass_from_counts(elem_rate, element_order=self.element_order) / 1000.0
            rel_mass_error = torch.abs(pred_mass - mass_tgt) / (mass_tgt + 1e-6)
            mass_penalty = rel_mass_error.mean()

            # Diversity penalty
            diversity_penalty = self._batch_diversity_penalty(elem_rate)

            # Ramp element loss
            lambda_elem = min(self.lambda_elem_max,
                              self.lambda_elem_start + (epoch / self.ramp_epochs) *
                              (self.lambda_elem_max - self.lambda_elem_start))

            loss = (lambda_elem * elem_loss +
                    self.lambda_mass_penalty * mass_penalty +
                    (self.lambda_presence * pres_loss if self.use_presence else 0.0) +
                    self.lambda_diversity * diversity_penalty +
                    self.lambda_entropy * ent_loss)

        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.scaler.step(self.optimizer)
        self.scaler.update()

        return {
            "elem_loss": elem_loss.item(),
            "mass_penalty": mass_penalty.item(),
            "pres_loss": pres_loss.item() if self.use_presence else 0.0,
            "diversity_penalty": diversity_penalty.item(),
            "entropy_loss": ent_loss.item(),
            "total_loss": loss.item()
        }