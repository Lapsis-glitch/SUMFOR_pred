# Fragmentation Rules & Enumerator Review

**Date:** 2026-04-14  
**Scope:** All rule packs in `fragmentation_rules_info/`, the enumerators
(`PeakDrivenEnumerator`, `HybridEnumerator`, `BDEDrivenEnumerator`,
`RecursiveFragmenter`), supporting modules (`chemistry.py`, `formula.py`,
`fps_scoring.py`), and the main pipeline (`Large_data.py`).

---

## A. Bugs / Correctness Issues

### A1. DBE formula omits phosphorus (`chemistry.py:62-76`)

The current RDBE formula is:

```
DBE = C − H/2 + N/2 + 1 − X/2
```

The standard IUPAC RDBE formula for CHONPS + halogens is:

```
DBE = C − H/2 + N/2 + P/2 + 1 − X/2
```

Phosphorus is trivalent (like nitrogen) and should contribute `+P/2`.
This means **every phosphorus-containing compound currently gets an
underestimated DBE**, which affects:
- functional group detection (aromatic, alkene triggers)
- complementary-loss filter (may wrongly accept bad fragments)
- FPS scoring via `_aromatic_stabilization` and `_cation_stability`

**Fix:** Add `P = elems.get("P", 0)` and include `+ P / 2.0` in the return.

---

### A2. Halogen pack generates duplicate fragments (`halogen.py:83-98`)

Rules 2 (X• radical loss) and 3 (C–X alpha cleavage) perform the
*exact same subtraction* (`subtract(parent, X)` for each halogen).
The comment acknowledges this:

> "Approximated by subtracting X• (same as radical loss) but labeled
> separately for interpretability."

This doubles the candidate count for every halogen-containing compound
without adding information. `RecursiveFragmenter` deduplicates by
formula string, but only across depths — **within a single depth level
the duplicates survive** into the `PeakDrivenEnumerator.lookup` and
inflate scoring.

**Fix:** Remove rule 3 entirely, or give it a structurally distinct
operation (e.g. subtract CX instead of just X).

---

### A3. `alcohol.py` and `ether.py` define identical `generate_beta_o` (`alcohol.py:26-41`, `ether.py:16-31`)

Both files define the same `BETA_O_LOSSES` list and an identical
`generate_beta_o` function. When both `fg["alcohol"]` and `fg["ether"]`
are True (extremely common, since any molecule with O > 0, H ≥ 2, C ≥ 2
triggers both), every β-O fragment is generated **twice** with the same
label `"beta_O_…"`.

**Fix:** Move `generate_beta_o` and `BETA_O_LOSSES` into a shared
module (e.g. `base.py`) and call it from one place — either `universal`
or a dedicated oxygen helper — rather than from both packs.

---

### A4. Rule-flag post-filter is effectively broken (`fragmentation_rules.py:54-61`)

```python
family = source.split("_")[0]   # → "neutral", "alcohol", "aromatic", …
```

But `Large_data.py` defines RULE_FLAGS with keys like
`"neutral_losses"`, `"alcohol_rules"`, `"carbonyl_rules"`, etc.

These will **never** match `source.split("_")[0]`, so the filter passes
everything through regardless of flag values. The flags are currently
all `True` so no harm is done, but if someone sets
`"alcohol_rules": False` it will have no effect.

**Fix:** Either align the RULE_FLAGS keys with what `split("_")[0]`
actually produces, or change the filter to use a proper mapping
(e.g. `{"alcohol_rules": ["alcohol_beta", "alcohol_dehydration", …]}`).

---

### A5. FPS rule-family priors never match (`fps_scoring.py:32-65 vs rule sources`)

`RULE_FAMILY_PRIORS` has short keys like `"neutral_loss"`, `"alpha"`,
`"tropylium"`. But actual `rule_source` strings from the packs are
things like:

- `"neutral_loss_C1H2O1"`
- `"alpha_cleavage_C1H3"`
- `"aromatic_tropylium"`
- `"alcohol_beta_cleavage_C2H5O1"`

The lookup `RULE_FAMILY_PRIORS.get(rule_source, 0.4)` will **always**
fall through to the default `0.4` because no exact match exists. All
those carefully tuned prior values are dead code.

**Fix:** Use prefix matching instead of exact lookup:

```python
prior = 0.4   # default
for key, val in RULE_FAMILY_PRIORS.items():
    if key in rule_source:
        prior = val
        break
```

Or refactor rule sources to expose a `family` tag separately.

---

### A6. BDE recursive fragmentation is single-level only (`bde_fragmenter.py:177`)

After breaking a bond, child fragment bond data is populated with:

```python
"bde": 999.0   # placeholder
```

Since `threshold` defaults to 120 kcal/mol, no child bonds will ever be
broken. The `recursive_fragment` function advertises `max_depth=8` but
in practice it only fragments one level deep.

**Fix:** Either:
- Run MACE-BDE inference on child Mol objects (expensive but correct), or
- Propagate the parent bond-data BDE values for surviving bonds (heuristic
  but much better than 999.0), or
- Document that BDE fragmentation is intentionally single-step and set
  `max_depth=1`.

---

## B. Functional Group Detection Issues (`base.py`)

The detection is purely compositional (no structural awareness), which
causes massive over-triggering:

| Group      | Trigger                 | False-positive examples                       |
|------------|-------------------------|-----------------------------------------------|
| alcohol    | O > 0 **and** H ≥ 2    | carboxylic acids, amides, esters, ethers      |
| carbonyl   | O > 0 **and** C ≥ 1    | ethanol, methanol, phenol                     |
| ether      | O > 0 **and** C ≥ 2    | carboxylic acids, alcohols, esters            |
| ester      | O ≥ 2 **and** C ≥ 2    | diols, diketones, quinones                    |
| aromatic   | DBE ≥ 4 **and** C ≥ 6  | cyclohexadiene, polycyclics without aromaticity|
| alkene     | DBE ≥ 1 **and** C ≥ 2  | any ring (cyclopropane DBE = 1)               |

This is partly by design (cast a wide net, let ML/scoring filter).
But it means **every oxygen-containing molecule fires 4–5 functional
group packs simultaneously**, generating a large candidate space and
adding noise.

**Suggestion:** When the HybridEnumerator is used and an RDKit Mol is
available, replace the compositional heuristics with actual substructure
(SMARTS) matching. E.g.:

```python
fg["alcohol"]  = mol.HasSubstructMatch(Chem.MolFromSmarts("[OX2H]"))
fg["carbonyl"] = mol.HasSubstructMatch(Chem.MolFromSmarts("[CX3]=[OX1]"))
fg["ether"]    = mol.HasSubstructMatch(Chem.MolFromSmarts("[OD2]([#6])[#6]"))
fg["ester"]    = mol.HasSubstructMatch(Chem.MolFromSmarts("[CX3](=O)[OX2]"))
```

Keep the compositional fallback for the rule-only path.

---

## C. Missing Fragmentation Rules

### C1. **HCN / HNC loss** (critical for nitrogen aromatics)

Loss of HCN (27 Da) is one of the most diagnostic EI fragmentations for
pyridines, pyrimidines, quinolines, indoles, and any N-heterocycle. It is
completely absent from the amine rule pack.

```python
# amine.py additions
HCN_LOSS = Formula({"H": 1, "C": 1, "N": 1})   # HCN  (27 Da)
```

Should fire when `fg["amine"]` is True and DBE ≥ 4 (suggesting an
N-heterocycle).

---

### C2. **NO and NO₂ losses** (nitro compounds)

Loss of NO• (30 Da) and NO₂• (46 Da) are the dominant primary
fragmentation for nitroaromatics (TNT, nitrobenzene) and nitro-alkanes.

```python
NO_LOSS  = Formula({"N": 1, "O": 1})        # 30 Da
NO2_LOSS = Formula({"N": 1, "O": 2})        # 46 Da
```

Could be added to `amine.py` or a new `nitro.py` pack, gated on N > 0
and O ≥ 1 (or O ≥ 2 for NO₂).

---

### C3. **Retro-Diels-Alder (RDA)**

One of the most important rearrangement pathways for six-membered rings
containing a double bond. Typical losses:

```python
RDA_LOSSES = [
    Formula({"C": 2, "H": 2}),   # C₂H₂  (26 Da) — acetylene
    Formula({"C": 2, "H": 4}),   # C₂H₄  (28 Da) — ethylene (already in neutral)
    Formula({"C": 3, "H": 4}),   # C₃H₄  (40 Da) — allene / propyne
    Formula({"C": 4, "H": 6}),   # C₄H₆  (54 Da) — butadiene
]
```

Should fire when aromatic or alkene is detected. The C₂H₂ and C₂H₄
losses already exist in `universal`, but C₃H₄ and C₄H₆ are missing.

---

### C4. **Cyclopentadienyl cation C₅H₅⁺** (m/z 65)

Extremely common in mass spectra of aromatics (formed via ring
contraction of phenyl + loss of neutral). Missing from `aromatic.py`.

```python
CYCLOPENTADIENYL = Formula({"C": 5, "H": 5})   # m/z 65
```

---

### C5. **Larger alkyl series for alpha-cleavage**

`ALPHA_CLEAVAGE_LOSSES` stops at C₂H₅ / C₂H₄. For long-chain
aliphatics (fatty acids, waxes, alkanes), the homologous series
continues:

```python
Formula({"C": 3, "H": 6}),   # C₃H₆
Formula({"C": 3, "H": 7}),   # C₃H₇•
Formula({"C": 4, "H": 8}),   # C₄H₈
Formula({"C": 4, "H": 9}),   # C₄H₉•
```

However, adding too many inflates the candidate space. A reasonable
compromise: add up to C₄ for primary alpha-cleavage, and let recursion
handle the rest.

---

### C6. **Amide-specific rules**

Amides are common in drug-like molecules and natural products. Missing
losses:

```python
AMIDE_LOSSES = [
    Formula({"C": 1, "H": 1, "N": 1, "O": 1}),  # CHNO  (43 Da) — isocyanate loss
    Formula({"C": 1, "H": 2, "N": 1, "O": 1}),  # CH₂NO (44 Da) — formamide loss
    Formula({"C": 1, "H": 3, "N": 1}),            # CH₃N  (29 Da) — methylamine loss
]
```

Gate on `fg["amine"]` **and** `fg["carbonyl"]` both True.

---

### C7. **CO loss from phenols and quinones**

While CO loss exists in `universal.py` neutral losses, it deserves a
higher prior for phenols (phenol → cyclopentadiene → C₅H₆⁺ at m/z 66)
and quinones. Consider adding it explicitly to `aromatic.py` with a
higher FPS prior.

---

### C8. **CN• radical loss** (nitriles)

Loss of CN• (26 Da) is diagnostic for nitrile-containing compounds.

```python
CN_LOSS = Formula({"C": 1, "N": 1})   # 26 Da
```

Gate on N > 0 and DBE ≥ 2.

---

### C9. **Thiophenyl / thiophene cations** (sulfur aromatics)

For thiophene-containing compounds, specific diagnostic ions:

```python
THIOPHENE_CATION = Formula({"C": 4, "H": 3, "S": 1})   # C₄H₃S⁺ (83 Da)
CHS_CATION       = Formula({"C": 1, "H": 1, "S": 1})   # CHS⁺  (45 Da)
```

Gate on `fg["sulfur"]` and `fg["aromatic"]`.

---

### C10. **Double / consecutive neutral losses** (explicit)

While recursion handles secondary fragmentation, the most common
two-step sequences should be pre-computed for depth-1 availability
(they get a shallower depth score bonus):

| Sequence       | Net loss     | Occurrence                  |
|----------------|--------------|-----------------------------|
| H₂O + CO       | C₁H₂O₂ (46) | phenols, carboxylic acids   |
| H₂O + H₂O     | H₄O₂   (36) | diols, sugars               |
| CO + CO        | C₂O₂   (56) | quinones                    |
| HCl + HCl      | H₂Cl₂ (72)  | dichlorides                 |
| CO₂ + H₂O     | C₁H₂O₃ (62) | carboxylic acids            |

---

### C11. **CHO⁺ (m/z 29) and C₂H₃O⁺ (m/z 43) as named cations**

These already exist in `universal.py` (COMMON_CATIONS) and `ester.py`,
but they should have **explicit high FPS priors** since m/z 29 (CHO⁺)
and m/z 43 (C₂H₃O⁺ / CH₃CO⁺) are among the most common peaks in EI
spectra.

---

## D. Structural / Architectural Improvements

### D1. Add early DBE validation in `base.subtract()`

Currently, fragments with negative DBE are generated freely and only
caught downstream by the complementary-loss filter. Adding a fast check
at generation time would shrink the candidate space significantly:

```python
def subtract(parent, loss, check_dbe=True):
    result = ...  # existing logic
    if result is None:
        return None
    if check_dbe and dbe(result) < -0.5:
        return None
    return result
```

---

### D2. Deduplicate within each rule pack

Multiple packs can emit the same fragment formula (e.g. `universal`
neutral loss CO₂ and `ester` alkoxy loss might coincidentally produce
the same formula). Add a `seen` set inside `registry.generate_fragments`:

```python
def generate_fragments(parent, fg):
    results = []
    seen = set()
    for pack in RULE_PACKS:
        frags = pack.generate(parent, fg)
        for frag, rule in frags:
            key = frag.to_string()
            if key not in seen:
                seen.add(key)
                results.append((frag, rule))
    return results
```

---

### D3. `Formula.to_string()` uses alphabetical, not Hill order

Hill order (C first, H second, then alphabetical) is the chemistry
standard. The current alphabetical ordering produces `Br1C6H5` instead
of `C6H5Br`. This doesn't break anything (deduplication works fine),
but makes human inspection harder.

---

### D4. SMARTS-based functional group detection when Mol is available

As discussed in §B — pass the RDKit Mol (when available) to
`detect_functional_groups` for structural-awareness:

```python
def detect_functional_groups(parent: Formula, mol=None) -> dict:
    if mol is not None:
        return _detect_from_mol(mol)
    return _detect_from_formula(parent)   # existing compositional logic
```

---

### D5. Expose a `rule_family` tag alongside `rule_source`

Currently rule sources are free-form strings like
`"alcohol_beta_cleavage_C2H5O1"`. Normalizing them with a structured
tag (e.g. `family="alcohol"`, `type="beta_cleavage"`) would fix both
the rule-flag filtering (A4) and the FPS prior lookup (A5) in one shot.

---

### D6. Consider ion-type awareness (OE⁺• vs EE⁺)

EI produces radical cations (odd-electron, OE⁺•). Fragmentation can
produce either:
- **Even-electron (EE⁺) cations** via loss of a radical (e.g. CH₃•)
- **Odd-electron (OE⁺•) radical cations** via loss of a neutral
  molecule (e.g. H₂O, CO)

The current system does not track electron parity. Adding an
`is_radical` flag to fragments would enable the
**even-electron rule** (EE⁺ ions do not fragment to OE⁺• ions), which
is a powerful chemical filter that could improve precision significantly.

---

## E. Summary of Priorities

All items have been implemented. See git diff for details.

| Priority | Issue          | Impact on precision | Status |
|----------|----------------|---------------------|--------|
| 🔴 High  | A5: FPS priors dead code | Medium — all priors default to 0.4 | ✅ Fixed — prefix matching |
| 🔴 High  | A1: DBE missing P | Medium — wrong DBE for P compounds | ✅ Fixed |
| 🔴 High  | C1: HCN loss   | High — misses key N-het fragments  | ✅ Added to amine.py |
| 🟡 Med   | A2: Halogen duplicates | Low — inflates candidates | ✅ Removed rule 3 |
| 🟡 Med   | A3: Duplicate β-O | Low — doubles candidates | ✅ Shared via base.py, called from universal.py |
| 🟡 Med   | A6: BDE single-level | Med — limits BDE value | ✅ Propagates parent BDE values |
| 🟡 Med   | C2: NO/NO₂ losses | Med — nitro compounds fail | ✅ Added to amine.py |
| 🟡 Med   | C3: RDA losses | Med — cyclics underserved | ✅ Added to aromatic.py |
| 🟡 Med   | C4: C₅H₅⁺ cation | Low — aromatic spectra | ✅ Added to aromatic.py |
| 🟡 Med   | D1: Early DBE check | Med — shrinks candidates | ✅ In base.subtract() |
| 🟡 Med   | D2: Dedup in registry | Low — reduces noise | ✅ In registry.py |
| 🟢 Low   | A4: Rule flags broken | None (all True currently) | ✅ Pack-level filtering in registry |
| 🟢 Low   | C5: Larger alkyl series | Low — recursion handles | ✅ Extended to C4 |
| 🟢 Low   | D3: Hill order | Cosmetic only | ✅ C first, H second |
| 🟢 Low   | D4: SMARTS detection | Med but large refactor | ✅ detect_functional_groups_from_mol() |
| 🟢 Low   | D6: OE/EE parity | High potential but big change | ✅ Even-electron rule in RecursiveFragmenter |

### Additional items implemented

| Item | Description | Location |
|------|-------------|----------|
| C6 | Amide losses (CHNO, CH₂NO, CH₃N) | amine.py |
| C7 | CO loss from aromatic compounds (higher prior) | aromatic.py |
| C8 | CN• radical loss (nitriles) | amine.py |
| C9 | Thiophene cations (C₄H₃S⁺, CHS⁺) | sulfur.py |
| C10 | Double neutral losses (H₂O+CO, 2×H₂O, 2×CO, CO₂+H₂O, CO+CH₂) | universal.py |
| D5 | Rule-family priors aligned with actual rule_source strings | fps_scoring.py |

