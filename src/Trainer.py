import torch
import torch.nn as nn
import torch.nn.functional as F


class CountTrainer:
    def __init__(self, model, optimizer, device, scaler,
                 lambda_elem_start=0.1, lambda_elem_max=0.5, lambda_mass=1e-4,
                 use_presence=True, ramp_epochs=10):
        """
        Trainer for Spectrum2Counts model.

        Args:
            model: Spectrum2Counts
            optimizer: torch optimizer
            device: torch.device
            scaler: GradScaler for mixed precision
            lambda_elem_start: starting weight for element loss
            lambda_elem_max: max weight for element loss
            lambda_mass: weight for auxiliary mass regression loss
            use_presence: whether to use presence head
            ramp_epochs: epochs to ramp element loss weight
        """
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.scaler = scaler

        # Loss weights
        self.lambda_elem_start = lambda_elem_start
        self.lambda_elem_max = lambda_elem_max
        self.lambda_mass = lambda_mass
        self.use_presence = use_presence
        self.ramp_epochs = ramp_epochs

        if self.use_presence:
            self.presence_loss_fn = nn.BCEWithLogitsLoss()

    def train_step(self, batch, elem_targets_all, mass_targets_all, batch_indices, epoch=0):
        """
        batch: (spectrum, _)  (tokens are ignored in count‑only mode)
        elem_targets_all: tensor (N, num_elements)
        mass_targets_all: tensor (N,)
        batch_indices: indices of current batch
        """
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        spectrum, _ = batch
        spectrum = spectrum.to(self.device)

        # Targets
        elem_tgt = elem_targets_all[batch_indices].to(self.device)
        elem_tgt_log = torch.log1p(elem_tgt.float())
        mass_tgt = mass_targets_all[batch_indices].to(self.device) / 1000.0

        with torch.amp.autocast("cuda"):
            if self.use_presence:
                elem_logits, mass_pred, presence_logits = self.model(spectrum)
            else:
                elem_logits, mass_pred = self.model(spectrum)
                presence_logits = None

            # --- Losses ---
            elem_loss = F.smooth_l1_loss(elem_logits, elem_tgt_log)
            mass_loss = F.mse_loss(mass_pred.squeeze(-1), mass_tgt)

            if self.use_presence:
                presence_tgt = (elem_tgt > 0).float()
                pres_loss = self.presence_loss_fn(presence_logits, presence_tgt)
            else:
                pres_loss = torch.tensor(0.0, device=self.device)

            # Ramp lambda_elem
            lambda_elem = min(self.lambda_elem_max,
                              self.lambda_elem_start + (epoch / self.ramp_epochs) *
                              (self.lambda_elem_max - self.lambda_elem_start))

            # Final loss
            loss = (lambda_elem * elem_loss +
                    self.lambda_mass * mass_loss +
                    (0.1 * pres_loss if self.use_presence else 0.0))

        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        return (elem_loss.item(),
                mass_loss.item(),
                pres_loss.item() if self.use_presence else 0.0,
                loss.item())