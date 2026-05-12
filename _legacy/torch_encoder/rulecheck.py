import re
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from src.data_extract import parse_msp

# -----------------------------
# Atomic properties
# -----------------------------
ATOMIC_MASS = {
    "H": 1.0078, "C": 12.0000, "N": 14.0031, "O": 15.9949,
    "F": 18.9984, "P": 30.9738, "S": 31.9721, "Cl": 34.9689,
    "Br": 78.9183, "I": 126.9045, "Si": 27.9769
}
VALENCE = {"H":1,"C":4,"N":3,"O":2,"F":1,"P":5,"S":6,"Cl":1,"Br":1,"I":1,"Si":4}

# -----------------------------
# Formula parsing
# -----------------------------
def parse_formula_dict(formula_str):
    elements = defaultdict(int)
    for elem, count in re.findall(r'([A-Z][a-z]*)(\d*)', formula_str):
        elements[elem] += int(count) if count else 1
    return dict(elements)

def compute_mass(fdict):
    return sum(ATOMIC_MASS.get(e,0)*c for e,c in fdict.items())

# -----------------------------
# Rules
# -----------------------------
def nitrogen_rule_ok(fdict):
    nominal_mass = round(compute_mass(fdict))
    n_count = fdict.get("N", 0)
    return (nominal_mass % 2 == n_count % 2)

def valence_ok(fdict):
    total_valence = sum(VALENCE.get(e,0)*c for e,c in fdict.items())
    return total_valence % 2 == 0 and total_valence >= 0

def relaxed_hc_rule(C, H):
    if C == 0:
        return H == 0
    if C <= 4:
        return H <= 2 * C + 2
    ratio = H / C
    return 0.2 <= ratio <= 3.5

def check_7GR(fdict):
    results = {}
    C = fdict.get("C",0); H = fdict.get("H",0)
    N = fdict.get("N",0); O = fdict.get("O",0)
    P = fdict.get("P",0); S = fdict.get("S",0)
    Si = fdict.get("Si",0)

    # 1. Element count restrictions
    results["element_count"] = (C <= 100 and H <= 200 and N <= 20 and O <= 30 and P <= 6 and S <= 6)

    # 2. Valence (already checked separately, but include here too)
    total_valence = sum(VALENCE.get(e,0)*c for e,c in fdict.items())
    results["valence_even"] = (total_valence % 2 == 0)

    # 3. Isotopic plausibility (stubbed)
    results["isotopic_pattern"] = True

    # 4. H/C ratio (relaxed)
    results["H/C"] = relaxed_hc_rule(C, H)

    # 5. Heteroatom/C ratios (relaxed N/C rule)
    if C > 0:
        if C <= 6:
            results["N/C"] = (N <= 4)
        else:
            results["N/C"] = (N/C <= 1.0 and N <= 14)
    else:
        results["N/C"] = (N == 0)

    results["O/C"] = (O <= C)
    results["P/C"] = (P <= C/3+1)
    results["S/C"] = (S <= C/2+1)

    # 6. Element ratio probabilities (simplified)
    results["element_ratio_prob"] = (N <= 20 and O <= 30 and S <= 6 and P <= 6)

    # 7. TMS check
    results["TMS_check"] = (Si == 0 or C > 0)

    return results

def check_all_rules(fdict):
    results = {}
    results["nitrogen_rule"] = nitrogen_rule_ok(fdict)
    results["valence_rule"] = valence_ok(fdict)
    results.update(check_7GR(fdict))
    return results

# -----------------------------
# Run checks + collect ratios
# -----------------------------
spectra = parse_msp("/home/rat/Leco/4Mix_comparison/4-Mix_Complete/NIST_mainlib/all.MSP")

violations = []
rule_counts = defaultdict(int)
ratios = {"H/C": [], "N/C": [], "O/C": []}

for spec in spectra:
    formula = spec.get("formula") or spec.get("Formula")
    if not formula:
        continue
    fdict = parse_formula_dict(formula)
    rules = check_all_rules(fdict)
    bad = [r for r,ok in rules.items() if not ok]
    for r in bad:
        rule_counts[r] += 1
    if bad:
        violations.append((formula, bad))

    # Collect ratios
    C = fdict.get("C",0)
    H = fdict.get("H",0)
    N = fdict.get("N",0)
    O = fdict.get("O",0)
    if C > 0:
        ratios["H/C"].append(H/C)
        ratios["N/C"].append(N/C)
        ratios["O/C"].append(O/C)

print(f"Checked {len(spectra)} entries")
print(f"Found {len(violations)} formulas violating at least one rule")

print("\nViolation counts per rule:")
for r,c in rule_counts.items():
    print(f"{r}: {c}")

print("\nExamples:")
for f,bad in violations[:20]:
    print(f"Formula {f} violates: {bad}")

# -----------------------------
# Plot distributions
# -----------------------------
sns.set(style="whitegrid")
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for ax, key in zip(axes, ["H/C", "N/C", "O/C"]):
    sns.histplot(ratios[key], bins=50, kde=False, ax=ax, color="steelblue")
    ax.set_title(f"{key} ratio distribution")
    ax.set_xlabel(key)
    ax.set_ylabel("Count")

plt.tight_layout()
plt.show()

# -----------------------------
# Optional: van Krevelen diagram (H/C vs O/C, colored by N/C)
# -----------------------------
plt.figure(figsize=(6,6))
plt.scatter(ratios["O/C"], ratios["H/C"],
            c=ratios["N/C"], cmap="viridis", s=10, alpha=0.7)
plt.colorbar(label="N/C ratio")
plt.xlabel("O/C")
plt.ylabel("H/C")
plt.title("Van Krevelen Diagram (colored by N/C)")
plt.show()