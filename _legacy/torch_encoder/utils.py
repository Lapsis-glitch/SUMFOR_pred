import numpy as np
from itertools import product

# --- Atomic masses (monoisotopic) ---
mass_table = {
    'C': 12.0000, 'H': 1.0078, 'O': 15.9949, 'N': 14.0031,
    'S': 31.9721, 'P': 30.9738, 'Cl': 34.9689, 'Br': 78.9183
}

# --- RDBE (rings + double bonds) ---
def rdbe(c, h, n, x=0):
    return c - h/2 + n/2 + 1 - x/2

# --- Fast mass calculator ---
def fast_mass(fdict):
    return sum(fdict[e] * mass_table[e] for e in fdict)

# --- Precursor selection with diagnostics ---
def find_precursor(mz, intensity, rel_threshold=0.05, verbose=False):
    if len(mz) == 0:
        if verbose: print("No peaks found in spectrum.")
        return None
    norm_int = intensity / intensity.max()
    mask = norm_int >= rel_threshold
    if not np.any(mask):
        precursor = mz[np.argmax(intensity)]
        if verbose: print(f"No peaks above {rel_threshold*100:.1f}% of base peak. "
                          f"Falling back to base peak at m/z {precursor}.")
        return precursor
    precursor = mz[mask].max()
    if verbose: print(f"Precursor chosen: m/z {precursor} "
                      f"(highest m/z above {rel_threshold*100:.1f}% of base peak).")
    return precursor

# --- Seven Golden Rules plausibility check (strict, for precursor only) ---
def plausible_7gr(formula_dict, precursor_mass=None, spectrum=None, verbose=False):
    C,H,O,N,S,P,Cl,Br = (formula_dict.get(e,0) for e in ['C','H','O','N','S','P','Cl','Br'])
    if C > 70 or H > 140 or O > 20 or N > 15 or S > 6 or P > 6 or Cl > 6 or Br > 6:
        return False
    if C == 0: return False
    if C > 0 and not (0.2 <= H / C <= 3.1): return False
    if precursor_mass is not None:
        nominal_mass = int(round(precursor_mass))
        if (nominal_mass % 2 == 0 and N % 2 != 0) or (nominal_mass % 2 == 1 and N % 2 != 1):
            return False
    rdbe_val = rdbe(C, H, N, x=(Cl + Br))
    if rdbe_val < 0 or rdbe_val > 40: return False
    if C > 0:
        if O / C > 1.2 or N / C > 1.3 or P / C > 0.3 or S / C > 0.8:
            return False
    return True

# --- Candidate generation (strict, for precursor) ---
def generate_candidates(precursor_mass, elements=None, tol=0.5, spectrum=None, verbose=False):
    if elements is None:
        elements = {'C':30,'H':60,'O':10,'N':5,'S':3,'P':2,'Cl':2,'Br':2}
    candidates = []
    ranges = [range(1, elements['C']+1),
              range(0, elements['H']+1),
              range(0, elements['O']+1),
              range(0, elements['N']+1),
              range(0, elements['S']+1),
              range(0, elements['P']+1),
              range(0, elements['Cl']+1),
              range(0, elements['Br']+1)]
    for c,h,o,n,s,p,cl,br in product(*ranges):
        fdict = {'C':c,'H':h,'O':o,'N':n,'S':s,'P':p,'Cl':cl,'Br':br}
        mass = (c*mass_table['C'] + h*mass_table['H'] + o*mass_table['O'] +
                n*mass_table['N'] + s*mass_table['S'] + p*mass_table['P'] +
                cl*mass_table['Cl'] + br*mass_table['Br'])
        if abs(mass - precursor_mass) > tol:
            continue
        if plausible_7gr(fdict, precursor_mass, spectrum, verbose=verbose):
            formula = ''.join(f"{e}{fdict[e]}" for e in fdict if fdict[e] > 0)
            candidates.append((formula, mass, fdict))
    return candidates

# --- Fragment candidate generation (relaxed rules, allow ions) ---
def generate_fragment_candidates(mz_val, elements=None, tol=0.5):
    if elements is None:
        elements = {'C':15,'H':30,'O':5,'N':3,'S':2,'P':2,'Cl':2,'Br':2}
    candidates = []
    ranges = [range(0, elements['C']+1),
              range(0, elements['H']+1),
              range(0, elements['O']+1),
              range(0, elements['N']+1),
              range(0, elements['S']+1),
              range(0, elements['P']+1),
              range(0, elements['Cl']+1),
              range(0, elements['Br']+1)]
    for c,h,o,n,s,p,cl,br in product(*ranges):
        if c+h+o+n+s+p+cl+br == 0:
            continue
        fdict = {'C':c,'H':h,'O':o,'N':n,'S':s,'P':p,'Cl':cl,'Br':br}
        mass = (c*mass_table['C'] + h*mass_table['H'] + o*mass_table['O'] +
                n*mass_table['N'] + s*mass_table['S'] + p*mass_table['P'] +
                cl*mass_table['Cl'] + br*mass_table['Br'])
        if abs(mass - mz_val) <= tol:
            formula = ''.join(f"{e}{fdict[e]}" for e in fdict if fdict[e] > 0)
            candidates.append((formula, mass, fdict))
    return candidates

# --- Fragment helpers ---
def select_top_peaks(mz, intensity, precursor_mz, N=5, rel_threshold=0.02):
    norm = intensity / intensity.max() if intensity.max() > 0 else intensity
    exclude_idx = np.where(mz == int(round(precursor_mz)))[0]
    mask = norm >= rel_threshold
    if len(exclude_idx) > 0:
        mask[exclude_idx[0]] = False
    idx = np.where(mask)[0]
    idx_sorted = idx[np.argsort(intensity[idx])[::-1]]
    chosen = idx_sorted[:N]
    return mz[chosen], intensity[chosen]

def subformula_possible(parent, fragment):
    for e in fragment:
        if fragment[e] > parent.get(e, 0):
            return False
    return True

def isotope_match_score(fdict, precursor_mass, spectrum):
    mz, intensity = spectrum
    base_idx = np.where(mz == int(round(precursor_mass)))[0]
    if len(base_idx) == 0:
        return 0
    I0 = intensity[base_idx[0]]
    if I0 == 0:
        return 0
    I1 = intensity[np.where(mz == int(round(precursor_mass))+1)[0][0]] if np.any(mz == int(round(precursor_mass))+1) else 0
    I2 = intensity[np.where(mz == int(round(precursor_mass))+2)[0][0]] if np.any(mz == int(round(precursor_mass))+2) else 0
    obs1, obs2 = I1/I0, I2/I0
    pred1, pred2 = isotope_pattern_estimate(fdict)
    score1 = max(0, 1 - abs(obs1 - pred1)/max(pred1,0.01))
    score2 = max(0, 1 - abs(obs2 - pred2)/max(pred2,0.01))
    return score1 + score2

# --- Main prediction function with fragment + isotope scoring ---
def predict_formula_from_spectrum(spec, converter, elements=None, tol=0.5,
                                  rel_threshold=0.05, topN=10, verbose=True,
                                  precursor_override=None):
    if elements is None:
        elements = {'C':30,'H':60,'O':10,'N':5,'S':3,'P':2,'Cl':2,'Br':2}
    # Convert spectrum to nominal mass
    if converter is None:
        mz_nom = spec.mz
        int_nom = spec.intensity
    else:
        mz_nom, int_nom = converter.convert(spec.mz, spec.intensity)
    if len(mz_nom) == 0:
        if verbose:
            print("No peaks after conversion.")
        return []

    # Use override if provided, otherwise detect precursor
    if precursor_override is not None:
        precursor_mass = precursor_override
        if verbose:
            print(f"Precursor override used: m/z {precursor_mass}")
    else:
        precursor_mass = find_precursor(mz_nom, int_nom, rel_threshold=rel_threshold, verbose=verbose)
        if precursor_mass is None:
            if verbose:
                print("No precursor could be determined.")
            return []

    # strict precursor candidates
    precursor_candidates = generate_candidates(precursor_mass, elements, tol,
                                               spectrum=(mz_nom, int_nom), verbose=verbose)

    # relaxed fragment candidates
    top_mz, top_int = select_top_peaks(mz_nom, int_nom, precursor_mass, N=topN)
    fragment_candidates = {mz: generate_fragment_candidates(mz, elements, tol)
                           for mz in top_mz}

    # Score precursor candidates by explained fragments + isotope match
    scored = []
    for formula, mass, fdict in precursor_candidates:
        explained = 0
        for mz, cand_list in fragment_candidates.items():
            for _, _, frag_dict in cand_list:
                if subformula_possible(fdict, frag_dict):
                    explained += 1
                    break  # count each fragment peak at most once

        iso_score = isotope_match_score(fdict, precursor_mass, (mz_nom, int_nom))
        composite = explained * 2 + iso_score - abs(mass - precursor_mass)
        scored.append((formula, mass, explained, iso_score, composite))

    ranked = sorted(scored, key=lambda x: -x[4])

    if verbose:
        print(f"Precursor m/z {precursor_mass}, {len(precursor_candidates)} candidates")
        for f, m, s, iso, comp in ranked[:10]:
            print(f"{f:12s} mass={m:.4f} explained={s} iso={iso:.2f} score={comp:.2f}")

    return ranked

# --- Isotope pattern estimation ---
def isotope_pattern_estimate(fdict):
    """
    Estimate expected M+1 and M+2 isotope ratios based on elemental composition.
    Returns: (M+1 ratio, M+2 ratio)
    """
    C = fdict.get('C', 0)
    H = fdict.get('H', 0)
    O = fdict.get('O', 0)
    N = fdict.get('N', 0)
    S = fdict.get('S', 0)
    P = fdict.get('P', 0)
    Cl = fdict.get('Cl', 0)
    Br = fdict.get('Br', 0)

    # Approximate isotope contributions
    m1 = C * 0.011 + N * 0.0037 + O * 0.0004  # M+1: mostly 13C, some 15N, 17O
    m2 = S * 0.044 + Cl * 0.32 + Br * 0.98 + O * 0.002  # M+2: 34S, 37Cl, 81Br, 18O

    return m1, m2
