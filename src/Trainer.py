import torch
import torch.nn.functional as F
from src.FormulaUtils import FormulaUtils


def expected_counts_from_logits(logits):
    counts_exp = {}
    for elem, lg in logits.items():
        probs = torch.softmax(lg, dim=-1)
        classes = torch.arange(lg.size(-1), device=lg.device).float()
        exp = (probs * classes.unsqueeze(0)).sum(dim=-1)
        counts_exp[elem] = exp
    return counts_exp


def batch_mass_from_expected_counts(counts_exp):
    device = next(iter(counts_exp.values())).device
    B = next(iter(counts_exp.values())).shape[0]
    mass = torch.zeros(B, device=device)
    for elem, exp in counts_exp.items():
        mass = mass + exp * FormulaUtils.ATOMIC_MASS.get(elem, 0.0)
    return mass


class SpectrumTrainer:
    """
    Trainer for MassSpectrumEncoder with:
      - per-element weighted CE loss
      - optional presence head loss
      - stronger relative mass penalty
    """

    def __init__(self, model, optimizer, device,
                 element_order,
                 class_weights=None,
                 lambda_presence=0.1,
                 lambda_entropy=0.01,
                 lambda_mass=0.5,   # stronger default
                 grad_clip=1.0):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.element_order = element_order
        # Default: emphasize C and H
        self.class_weights = class_weights or {"C": 2.0, "H": 2.0}
        self.lambda_presence = lambda_presence
        self.lambda_entropy = lambda_entropy
        self.lambda_mass = lambda_mass
        self.grad_clip = grad_clip

    def set_class_weights(self, weights: dict):
        self.class_weights = weights

    def train_step(self, batch, mass_targets=None):
        self.model.train()
        self.optimizer.zero_grad()

        mz, intens, mask, counts = batch
        mz, intens, mask, counts = mz.to(self.device), intens.to(self.device), mask.to(self.device), counts.to(self.device)

        outputs = self.model(mz, intens, mask)
        if isinstance(outputs, tuple):
            predictions, presence_logits = outputs
            presence_tgt = (counts > 0).float()
            bce = F.binary_cross_entropy_with_logits(presence_logits, presence_tgt)
            p = torch.sigmoid(presence_logits).clamp(1e-6, 1 - 1e-6)
            entropy = -(p * torch.log(p) + (1 - p) * torch.log(1 - p)).mean()
            presence_loss = self.lambda_presence * bce + self.lambda_entropy * entropy
        else:
            predictions = outputs
            presence_loss = torch.tensor(0.0, device=self.device)

        # Weighted element classification loss
        ce_total = 0.0
        for e_idx, e in enumerate(self.element_order):
            logits = predictions[e]
            targets = counts[:, e_idx]
            if e in self.class_weights:
                ce = F.cross_entropy(logits, targets) * self.class_weights[e]
            else:
                ce = F.cross_entropy(logits, targets)
            ce_total = ce_total + ce

        # Relative mass penalty
        if mass_targets is not None:
            counts_exp = expected_counts_from_logits(predictions)
            pred_mass = batch_mass_from_expected_counts(counts_exp)
            mass_tgt = mass_targets.to(self.device)
            rel_err = torch.abs(pred_mass - mass_tgt) / (mass_tgt + 1e-6)
            mass_penalty = rel_err.mean()
        else:
            mass_penalty = torch.tensor(0.0, device=self.device)

        loss = ce_total + presence_loss + self.lambda_mass * mass_penalty
        loss.backward()
        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.optimizer.step()

        return {
            "elem_ce": ce_total.item(),
            "presence": presence_loss.item() if torch.is_tensor(presence_loss) else 0.0,
            "mass_penalty": mass_penalty.item(),
            "total": loss.item()
        }

    def validate(self, val_loader):
        self.model.eval()
        elem_correct = {e: 0 for e in self.element_order}
        elem_total = {e: 0 for e in self.element_order}

        with torch.no_grad():
            for mz, intens, mask, counts in val_loader:
                mz, intens, mask, counts = mz.to(self.device), intens.to(self.device), mask.to(self.device), counts.to(self.device)
                outputs = self.model(mz, intens, mask)
                predictions = outputs[0] if isinstance(outputs, tuple) else outputs

                for e_idx, e in enumerate(self.element_order):
                    logits = predictions[e]
                    pred_class = logits.argmax(dim=-1)
                    true_class = counts[:, e_idx]
                    elem_correct[e] += (pred_class == true_class).sum().item()
                    elem_total[e] += true_class.numel()

        accs = {e: elem_correct[e] / max(1, elem_total[e]) for e in self.element_order}
        return accs