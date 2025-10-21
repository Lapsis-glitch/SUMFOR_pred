import os
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

from dataset import SpectrumFormulaDataset, FormulaTokenizer
from model import Spectrum2Formula   # hybrid version with use_formula_transformer=True
from src.DatasetBuilder import DatasetBuilder
from src.FormulaUtils import FormulaUtils
from src.Trainer import Trainer
from src.BeamDecoder import BeamSearchDecoder
from src.data_extract import parse_msp


# -----------------------------
# Candidate sampler factory
# -----------------------------
def candidate_sampler_factory(all_elem_targets, mass_targets, mass_tol=0.5, num_neg=15):
    """
    Returns a function that samples negatives for a given index.
    - all_elem_targets: tensor (N, E) element counts
    - mass_targets: tensor (N,) precursor masses
    - mass_tol: deviation in Da for negatives
    - num_neg: number of negatives per sample
    """
    N, E = all_elem_targets.size()

    def sampler(idx):
        true_mass = mass_targets[idx].item()
        # indices with similar mass but different formula
        candidates = [j for j in range(N)
                      if j != idx and abs(mass_targets[j].item() - true_mass) < mass_tol]
        if len(candidates) < num_neg:
            candidates = random.sample(range(N), num_neg)
        else:
            candidates = random.sample(candidates, num_neg)
        return all_elem_targets[candidates]  # (num_neg, E)

    return sampler


# -----------------------------
# Helper: convert decoded formulas to count tensors
# -----------------------------
def formulas_to_counts(formulas, element_order):
    """
    Convert list of formula strings to (K, E) count tensor.
    """
    counts = []
    for f in formulas:
        fdict = FormulaUtils.parse_formula_dict(f)
        counts.append([fdict.get(e, 0) for e in element_order])
    return torch.tensor(counts, dtype=torch.float32)


def main():
    # --- Load spectra ---
    spectra = parse_msp("/home/rat/Leco/4Mix_comparison/4-Mix_Complete/NIST_mainlib/all.MSP")
    print(f"Parsed {len(spectra)} entries")

    # --- Build dataset (CHONS only) ---
    tokenizer = FormulaTokenizer()  # multi-digit counts tokenizer
    builder = DatasetBuilder(spectra, tokenizer, max_mz=1000, max_len=30,
                             allowed_elements={"C", "H", "O", "N", "S"})
    filtered_spectra = builder.filter_spectra()
    print(f"Filtered dataset (CHONS only): {len(filtered_spectra)} valid formulas out of {len(spectra)}")

    ELEMENT_ORDER = ["C", "H", "N", "O", "S"]
    num_elements = len(ELEMENT_ORDER)

    elem_targets_all, mass_targets_all = builder.build_targets(filtered_spectra, ELEMENT_ORDER)
    dataset = SpectrumFormulaDataset(filtered_spectra, tokenizer, max_mz=1000, max_len=30)

    # --- Train/val split ---
    indices = list(range(len(dataset)))
    train_idx, val_idx = train_test_split(indices, test_size=0.1, random_state=42)

    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=256, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=1, shuffle=False,
                            num_workers=2, pin_memory=True)

    # --- Model + Optimizer ---
    model = Spectrum2Formula(
        input_dim=1000,
        vocab_size=len(tokenizer.vocab),
        embed_dim=256,
        n_layers=4,
        n_heads=4,
        ff_dim=1024,
        num_elements=num_elements,
        use_presence_head=True,
        use_formula_transformer=True  # enable formula scoring head
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Different LR for element head vs rest
    elem_params = list(model.elem_head.parameters())
    other_params = [p for n, p in model.named_parameters() if not any(ep is p for ep in elem_params)]
    optimizer = torch.optim.Adam([
        {"params": other_params, "lr": 1e-4},
        {"params": elem_params, "lr": 5e-4}
    ])
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.stoi["<pad>"])
    scaler = torch.amp.GradScaler("cuda")

    # --- Resume from checkpoint if available ---
    start_epoch = 0
    best_val_acc = 0.0
    checkpoint_path = "checkpoint_epoch_150.pt"
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint.get("epoch", 0) + 1
        best_val_acc = checkpoint.get("val_acc", 0.0)
        print(f"Resumed from checkpoint at epoch {start_epoch}, "
              f"Val Acc {best_val_acc:.3f}, "
              f"Elem Acc {checkpoint.get('val_elem_acc', 0):.3f}, "
              f"Loss {checkpoint.get('loss', 0):.4f}")

    # --- Trainer + Decoder ---
    trainer = Trainer(
        model, optimizer, criterion, device, scaler,
        lambda_elem_start=0.1, lambda_elem_max=0.5,
        lambda_mass=1e-4, use_presence=True, ramp_epochs=10,
        lambda_consistency=0.01, lambda_seq_mass=1e-4,
        lambda_formula_rank=1e-3  # weight for ranking loss
    )

    decoder = BeamSearchDecoder(
        tokenizer,
        penalty_weight=1.0,
        mass_tol=0.3,
        repetition_penalty=1.2,
        length_penalty=1.0,
        no_repeat_ngram=2
    )

    # --- Candidate sampler ---
    candidate_sampler = candidate_sampler_factory(elem_targets_all, mass_targets_all,
                                                  mass_tol=0.5, num_neg=15)

    # --- Training loop ---
    num_epochs = 150
    K = 5

    for epoch in range(start_epoch, num_epochs):
        total_ce = total_elem = total_mass = total_pres = total_cons = total_seq_mass = total_rank = total_loss = 0.0

        for batch_i, (spectrum, tokens) in enumerate(train_loader):
            spectrum = spectrum.to(device)
            tokens = tokens.to(device)

            start = batch_i * train_loader.batch_size
            end = start + spectrum.size(0)
            batch_indices = train_idx[start:end]

            ce_l, elem_l, mass_l, pres_l, cons_l, seq_mass_l, rank_l, loss = trainer.train_step(
                (spectrum, tokens),
                elem_targets_all, mass_targets_all,
                batch_indices,
                epoch=epoch,
                tokenizer=tokenizer,
                element_order=ELEMENT_ORDER,
                candidate_sampler=candidate_sampler
            )

            total_ce += ce_l
            total_elem += elem_l
            total_mass += mass_l
            total_pres += pres_l
            total_cons += cons_l
            total_seq_mass += seq_mass_l
            total_rank += rank_l
            total_loss += loss

        avg_ce = total_ce / len(train_loader)
        avg_elem = total_elem / len(train_loader)
        avg_mass = total_mass / len(train_loader)
        avg_pres = total_pres / len(train_loader)
        avg_cons = total_cons / len(train_loader)
        avg_seq_mass = total_seq_mass / len(train_loader)
        avg_rank = total_rank / len(train_loader)
        avg_loss = total_loss / len(train_loader)

        # --- Validation with optional re-ranking ---
        correct_top1, total, avg_conf = 0, 0, 0.0
        elem_correct_top1, elem_total_top1 = 0, 0
        hitk_count = 0
        elem_correct_topk, elem_total_topk = 0, 0

        for i, (spectrum, tokens) in enumerate(val_loader):
            spectrum = spectrum.to(device)
            true_formula = tokenizer.decode(tokens[0].tolist())

            # Beam decode candidates
            preds = decoder.decode(
                model, spectrum[0], device=device,
                precursor_mass=None, formula_utils=FormulaUtils,
                return_k=K,
                element_order=ELEMENT_ORDER
            )

            # Optional re-ranking with formula head
            # Build (K, E) counts tensor and score with model.score_formulas
            candidate_formulas = [f for f, _ in preds]
            candidate_counts = formulas_to_counts(candidate_formulas, ELEMENT_ORDER).unsqueeze(0)  # (1, K, E)
            candidate_counts = torch.log1p(candidate_counts).to(device)
            energies = model.score_formulas(spectrum, candidate_counts)  # (1, K)
            reranked = sorted(zip(candidate_formulas, energies.squeeze(0).tolist()),
                              key=lambda x: x[1], reverse=True)
            best_formula, best_energy = reranked[0]
            avg_conf += best_energy  # energy as confidence proxy

            if best_formula == true_formula:
                correct_top1 += 1
            total += 1

            c1, t1 = FormulaUtils.element_accuracy(true_formula, best_formula)
            elem_correct_top1 += c1
            elem_total_top1 += t1

            # Hit@K: any exact match among beams (pre-rerank for strictness)
            if any(f == true_formula for f, _ in preds):
                hitk_count += 1

            # Element accuracy@K: best candidate across K (post-rerank doesn't change max achievable)
            best_c, best_t = 0, 0
            for f, _ in preds:
                c, t = FormulaUtils.element_accuracy(true_formula, f)
                if c > best_c:
                    best_c, best_t = c, t
            elem_correct_topk += best_c
            elem_total_topk += best_t

            if i < 5:
                print(f"[Val sample {i}] True: {true_formula}")
                for rank, (formula, energy) in enumerate(reranked[:K], 1):
                    print(f"   Top{rank}: {formula} | Score: {energy:.3f}")

        acc_top1 = correct_top1 / total if total > 0 else 0.0
        elem_acc_top1 = elem_correct_top1 / elem_total_top1 if elem_total_top1 > 0 else 0.0
        hit_at_k = hitk_count / total if total > 0 else 0.0
        elem_acc_topk = elem_correct_topk / elem_total_topk if elem_total_topk > 0 else 0.0
        avg_conf /= max(total, 1)

        print(
            f"Epoch {epoch+1}, Loss {avg_loss:.4f} "
            f"(CE {avg_ce:.4f}, Elem {avg_elem:.4f}, Mass {avg_mass:.4f}, "
            f"Pres {avg_pres:.4f}, Cons {avg_cons:.4f}, SeqMass {avg_seq_mass:.4f}, Rank {avg_rank:.4f}), "
            f"Val Acc@1 {acc_top1:.3f}, Elem Acc@1 {elem_acc_top1:.3f}, "
            f"Hit@{K} {hit_at_k:.3f}, Elem Acc@{K} {elem_acc_topk:.3f}, "
            f"Avg Score {avg_conf:.3f}"
        )

        # Save best model by Acc@1
        if acc_top1 > best_val_acc:
            best_val_acc = acc_top1
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_loss,
                "val_acc": acc_top1,
                "val_elem_acc": elem_acc_top1,
                "hit_at_k": hit_at_k,
                "elem_acc_topk": elem_acc_topk,
            }, "best_model.pt")
            print(f"✅ Saved best model at epoch {epoch + 1}")

        # Periodic checkpointing
        if (epoch + 1) % 10 == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_loss,
                "val_acc": acc_top1,
                "val_elem_acc": elem_acc_top1,
                "hit_at_k": hit_at_k,
                "elem_acc_topk": elem_acc_topk,
            }, f"checkpoint_epoch_{epoch + 1}.pt")
            print(f"💾 Checkpoint saved at epoch {epoch + 1}")


if __name__ == "__main__":
    main()