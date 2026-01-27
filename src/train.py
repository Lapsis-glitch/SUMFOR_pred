import os
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from torch import optim

from dataset import MassSpectrumDataset, collate_batch
from model import MassSpectrumEncoder
from Trainer import SpectrumTrainer
from src.data_extract import parse_msp
from src.DatasetBuilder import DatasetBuilder
from src.FormulaUtils import FormulaUtils


def counts_to_formula(counts, element_order):
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
    builder = DatasetBuilder(
        spectra, None, max_mz=1000, max_len=30,
        allowed_elements={"C", "H", "O", "N", "S"}
    )
    filtered_spectra = builder.filter_spectra()
    print(f"Filtered dataset (CHONS only): {len(filtered_spectra)} valid formulas out of {len(spectra)}")

    ELEMENT_ORDER = ["C", "H", "N", "O", "S"]

    # Restrict to formulas with 3–6 carbons
    restricted_spectra = []
    for s in filtered_spectra:
        fdict = FormulaUtils.parse_formula_dict(s["Formula"])
        c_count = fdict.get("C", 0)
        if 3 <= c_count <= 6:
            restricted_spectra.append(s)
    print(f"Restricted dataset (C=3–6 only): {len(restricted_spectra)} entries")

    # Build MassSpectrumDataset (mz + intensities + counts)
    formulas = [FormulaUtils.parse_formula_dict(s["Formula"]) for s in restricted_spectra]
    spectra_peaks = [(s["mz"], s["intensities"]) for s in restricted_spectra]

    dataset = MassSpectrumDataset(
        spectra_peaks,
        formulas,
        ELEMENT_ORDER,
        max_peaks=500,
        max_mz=1000.0
    )

    # --- Train/val split ---
    indices = list(range(len(dataset)))
    train_idx, val_idx = train_test_split(indices, test_size=0.1, random_state=42)

    train_loader = DataLoader(
        Subset(dataset, train_idx), batch_size=64, shuffle=True,
        num_workers=4, pin_memory=True, collate_fn=collate_batch
    )
    val_loader = DataLoader(
        Subset(dataset, val_idx), batch_size=1, shuffle=False,
        num_workers=2, pin_memory=True, collate_fn=collate_batch
    )

    # --- Model ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    element_max_counts = {"C": 60, "H": 120, "N": 20, "O": 40, "S": 10}
    model = MassSpectrumEncoder(
        d_model=512, nhead=16, num_layers=6, dim_feedforward=1024,
        max_peaks=1000, element_max_counts=element_max_counts,
        use_presence_head=True, use_presence_gating=True
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-4)

    trainer = SpectrumTrainer(
        model, optimizer, device,
        element_order=ELEMENT_ORDER,
        class_weights=None,   # will be set dynamically
        lambda_presence=0.1, lambda_entropy=0.01, lambda_mass=0.1
    )

    # --- Checkpointing ---
    best_val_acc = 0.0
    checkpoint_path = "checkpoint_mass_encoder1.pt"
    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        best_val_acc = ckpt.get("val_acc", 0.0)
        print(f"Resumed from checkpoint with Val Acc {best_val_acc:.3f}")

    # --- Training loop ---
    num_epochs = 50
    for epoch in range(num_epochs):
        # Dynamically set element weights each epoch
        element_weights = {
            "C": 3.0 + 0.1 * epoch,  # ramp up carbon weight
            "H": 2.0,
            "N": 1.0,
            "O": 1.0,
            "S": 1.0
        }
        trainer.set_class_weights(element_weights)

        # Train
        for batch in train_loader:
            _, _, _, counts = batch
            true_mass = torch.tensor([
                FormulaUtils.compute_mass({e: int(c.item()) for e, c in zip(ELEMENT_ORDER, counts_row)})
                for counts_row in counts
            ], dtype=torch.float32)
            losses = trainer.train_step(batch, mass_targets=true_mass)

        # Validate
        accs = trainer.validate(val_loader)
        avg_acc = sum(accs.values()) / len(accs)
        print(f"Epoch {epoch+1}: Loss {losses['total']:.4f}, Val Acc {avg_acc:.3f}")
        print(f"    Weights used: {element_weights}")
        for e, a in accs.items():
            print(f"    {e}: {a:.3f}")

        # Print a few truth vs preds
        model.eval()
        with torch.no_grad():
            for i, (mz, intens, mask, counts) in enumerate(val_loader):
                mz, intens, mask = mz.to(device), intens.to(device), mask.to(device)
                outputs = model(mz, intens, mask)
                predictions = outputs[0] if isinstance(outputs, tuple) else outputs
                pred_counts = [predictions[e].argmax(dim=-1).item() for e in ELEMENT_ORDER]
                true_counts = counts[0].tolist()
                print(f"Val sample {i}: True {counts_to_formula(true_counts, ELEMENT_ORDER)} | "
                      f"Pred {counts_to_formula(pred_counts, ELEMENT_ORDER)}")
                if i >= 2:
                    break

        # Save best model
        if avg_acc > best_val_acc:
            best_val_acc = avg_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": avg_acc,
            }, "best_mass_encoder.pt")
            print(f"✅ Saved new best model at epoch {epoch+1}")

        # Periodic checkpoint
        if (epoch + 1) % 5 == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": avg_acc,
            }, checkpoint_path)
            print(f"💾 Checkpoint saved at epoch {epoch+1}")


if __name__ == "__main__":
    main()