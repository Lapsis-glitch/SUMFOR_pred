import torch

from src.FormulaUtils import FormulaUtils


class DatasetBuilder:
    def __init__(self, spectra, tokenizer, max_mz=1000, max_len=30, allowed_elements=None):
        self.spectra = spectra
        self.tokenizer = tokenizer
        self.max_mz = max_mz
        self.max_len = max_len
        self.allowed_elements = allowed_elements or {"C","H","N","O","S"}

    def filter_spectra(self):
        filtered = []
        for spec in self.spectra:
            formula = spec.get("formula") or spec.get("Formula")
            if not formula: continue
            fdict = FormulaUtils.parse_formula_dict(formula)
            if all(FormulaUtils.check_all_rules(fdict).values()) and all(e in self.allowed_elements for e in fdict):
                filtered.append(spec)
        return filtered

    def build_targets(self, specs, element_order):
        elem_targets, mass_targets = [], []
        for spec in specs:
            formula = spec.get("formula") or spec.get("Formula")
            fdict = FormulaUtils.parse_formula_dict(formula)
            counts = [fdict.get(e, 0) for e in element_order]
            elem_targets.append(counts)
            precursor = spec.get("precursor_mz") or spec.get("PrecursorMZ") or FormulaUtils.compute_mass(fdict)
            mass_targets.append(precursor)
        return torch.tensor(elem_targets, dtype=torch.float32), torch.tensor(mass_targets, dtype=torch.float32)