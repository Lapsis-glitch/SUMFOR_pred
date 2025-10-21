import os
import torch
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from sklearn.model_selection import train_test_split
import collections

from dataset import SpectrumCountDataset
from model import Spectrum2Counts
from src.DatasetBuilder import DatasetBuilder
from src.FormulaUtils import FormulaUtils
from src.CountTrainer import CountTrainer
from src.data_extract import parse_msp


def counts_to_formula(counts, element_order):
    """Convert a count vector into a formula string, skipping zero counts."""
    parts = []
    for elem, c in zip(element_order, counts):
        c = int(c)
        if c > 0:
            parts.append(elem)
            if c > 1:
                parts.append(str(c))
    return "".join(parts) if parts else "EMPTY"


def main():
    # --- Load spectra ---
    spectra = parse_msp("/home/rat/Leco/4Mix_comparison/4-Mix_Complete/NIST_mainlib/all.MSP")
    print(f"Parsed {len(spectra)} entries")

    # --- Build dataset (CHONS only) ---
    builder = DatasetBuilder(spectra, None, max_mz=1000, max_len=30,
                             allowed_elements={"C", "H", "O", "N", "S"})
    filtered_spectra = builder.filter_spectra()
    print(f"Filtered dataset (CHONS only): {len(filtered_spectra)} valid formulas out of {len(spectra)}")

    ELEMENT_ORDER = ["C", "H", "N", "O", "S"]
    num_elements = len(ELEMENT_ORDER)

    elem_targets_all, mass_targets_all = builder.build_targets(filtered_spectra, ELEMENT_ORDER)
    dataset = SpectrumCountDataset(filtered_spectra, ELEMENT_ORDER, max_mz=1000)

    # --- Train/val split ---
    indices = list(range(len(dataset)))
    train_idx, val_idx = train_test_split(indices, test_size=0.1, random_state=42)

    # --- Build weights for rebalancing (based on carbon count) ---
    carbon_counts = [FormulaUtils.parse_formula_dict(filtered_spectra[i]["Formula"]).get("C", 0)
                     for i in train_idx]
    freq = collections.Counter(carbon_counts)
    weights = [1.0 / freq[c] for c in carbon_counts]

    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=256,
                              sampler=sampler, num_workers=4, pin_memory=True)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=1, shuffle=False,
                            num_workers=2, pin_memory=True)

    # --- Model + Optimizer ---
    model = Spectrum2Counts(
        input_dim=1000,
        embed_dim=256,
        n_layers=4,
        n_heads=4,
        ff_dim=1024,
        num_elements=num_elements,
        use_presence_head=True
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda")

    # --- Scheduler: cosine with warmup ---
    total_steps = len(train_loader) * 150
    warmup_steps = max(500, len(train_loader) * 3)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.1415926535))).item()

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    # --- Resume from checkpoint if available ---
    start_epoch = 0
    best_val_acc = 0.0
    checkpoint_path = "checkpoint_counts.pt"
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint.get("epoch", 0) + 1
        best_val_acc = checkpoint.get("val_acc", 0.0)
        print(f"Resumed from checkpoint at epoch {start_epoch}, "
              f"Val Acc {best_val_acc:.3f}, "
              f"Loss {checkpoint.get('loss', 0):.4f}")

    # --- Trainer ---
    trainer = CountTrainer(
        model, optimizer, device, scaler,
        lambda_elem_start=0.2, lambda_elem_max=0.8,
        lambda_mass_penalty=0.2,
        lambda_presence=0.1,
        lambda_diversity=0.02,
        lambda_entropy=0.01,
        use_presence=True, ramp_epochs=10,
        element_order=ELEMENT_ORDER,
        element_weights=[1.0, 1.0, 2.0, 2.0, 3.0]  # emphasize N/O/S
    )

    # --- Training loop ---
    num_epochs = 150

    for epoch in range(start_epoch, num_epochs):
        total_elem = total_mass_pen = total_pres = total_div = total_ent = total_loss = 0.0

        for batch_i, (spectrum, counts) in enumerate(train_loader):
            spectrum, counts = spectrum.to(device), counts.to(device)

            # batch_indices must align with train_idx
            start = batch_i * train_loader.batch_size
            end = start + spectrum.size(0)
            batch_indices = train_idx[start:end]

            losses = trainer.train_step(
                (spectrum, counts),
                elem_targets_all, mass_targets_all,
                batch_indices,
                epoch=epoch
            )

            total_elem += losses["elem_loss"]
            total_mass_pen += losses["mass_penalty"]
            total_pres += losses["pres_loss"]
            total_div += losses["diversity_penalty"]
            total_ent += losses["entropy_loss"]
            total_loss += losses["total_loss"]

            scheduler.step()

        avg_elem = total_elem / len(train_loader)
        avg_mass_pen = total_mass_pen / len(train_loader)
        avg_pres = total_pres / len(train_loader)
        avg_div = total_div / len(train_loader)
        avg_ent = total_ent / len(train_loader)
        avg_loss = total_loss / len(train_loader)

        # --- Validation ---
        elem_correct, elem_total = 0, 0
        per_elem_correct = {e: 0 for e in ELEMENT_ORDER}
        per_elem_total = {e: 0 for e in ELEMENT_ORDER}
        sample_prints = 0

        for spectrum, counts in val_loader:
            spectrum, counts = spectrum.to(device), counts.to(device)
            with torch.no_grad():
                outputs = model(spectrum)
                if trainer.use_presence:
                    elem_pred, presence_logits = outputs
                else:
                    elem_pred = outputs

                pred_counts = elem_pred.round().clamp(min=0).int().cpu().numpy()[0]
                true_counts = counts.cpu().numpy()[0]

                for elem, pc, tc in zip(ELEMENT_ORDER, pred_counts, true_counts):
                    if pc == tc:
                        elem_correct += 1
                        per_elem_correct[elem] += 1
                    per_elem_total[elem] += 1
                    elem_total += 1

                if sample_prints < 5:
                    pred_formula = counts_to_formula(pred_counts, ELEMENT_ORDER)
                    true_formula = counts_to_formula(true_counts, ELEMENT_ORDER)
                    print(f"[Val sample {sample_prints}] True: {true_formula} | Pred: {pred_formula}")
                    sample_prints += 1

        elem_acc = elem_correct / elem_total if elem_total > 0 else 0.0

        print(
            f"Epoch {epoch+1}, Loss {avg_loss:.4f} "
            f"(Elem {avg_elem:.4f}, MassPen {avg_mass_pen:.4f}, "
            f"Pres {avg_pres:.4f}, Div {avg_div:.4f}, Ent {avg_ent:.4f}), "
            f"Val Elem Acc {elem_acc:.3f}"
        )
        for elem in ELEMENT_ORDER:
            acc = per_elem_correct[elem] / per_elem_total[elem] if per_elem_total[elem] > 0 else 0.0
            print(f"    {elem} Acc: {acc:.3f}")

        # Save best model
        if elem_acc > best_val_acc:
            best_val_acc = elem_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_loss,
                "val_acc": elem_acc,
            }, "best_model_counts.pt")
            print(f"✅ Saved best model at epoch {epoch + 1}")

        # Periodic checkpointing
        if (epoch + 1) % 10 == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_loss,
                "val_acc": elem_acc,
            }, f"checkpoint_counts_epoch_{epoch + 1}.pt")
            print(f"💾 Checkpoint saved at epoch {epoch + 1}")

if __name__ == "__main__":
    main()

