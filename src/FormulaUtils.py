import re
from collections import defaultdict
import torch


class FormulaUtils:
    ATOMIC_MASS = {
        "H": 1.0078, "C": 12.0000, "N": 14.0031, "O": 15.9949,
        "F": 18.9984, "P": 30.9738, "S": 31.9721, "Cl": 34.9689,
        "Br": 78.9183, "I": 126.9045, "Si": 27.9769
    }
    VALENCE = {
        "H": 1, "C": 4, "N": 3, "O": 2, "F": 1,
        "P": 5, "S": 6, "Cl": 1, "Br": 1, "I": 1, "Si": 4
    }

    @staticmethod
    def parse_formula_dict(formula_str: str):
        """Parse a formula string into a dict of element -> count."""
        elements = defaultdict(int)
        for elem, count in re.findall(r'([A-Z][a-z]*)(\d*)', formula_str):
            elements[elem] += int(count) if count else 1
        return dict(elements)

    @classmethod
    def compute_mass(cls, fdict: dict) -> float:
        """Compute monoisotopic mass from element counts (dict)."""
        return sum(cls.ATOMIC_MASS.get(e, 0) * c for e, c in fdict.items())

    @classmethod
    def compute_mass_from_counts(cls, counts, element_order):
        """
        Compute molecular mass from element count vectors.

        Args:
            counts: torch.Tensor or numpy array of shape (E,) or (B, E)
                    containing element counts (not log1p).
            element_order: list of element symbols in the same order as counts.

        Returns:
            torch.Tensor of shape (B,) with masses if input is 2D,
            or scalar tensor if input is 1D.
        """
        if not isinstance(counts, torch.Tensor):
            counts = torch.tensor(counts, dtype=torch.float32)

        if counts.dim() == 1:
            counts = counts.unsqueeze(0)

        masses = torch.tensor([cls.ATOMIC_MASS[e] for e in element_order],
                              device=counts.device, dtype=counts.dtype)
        return (counts * masses).sum(dim=1)

    @staticmethod
    def relaxed_hc_rule(C: int, H: int) -> bool:
        """Relaxed hydrogen/carbon ratio rule."""
        if C == 0:
            return H == 0
        if C <= 4:
            return H <= 2 * C + 2
        ratio = H / C
        return 0.2 <= ratio <= 3.5

    @classmethod
    def check_all_rules(cls, fdict: dict) -> dict:
        """Apply heuristic chemical plausibility rules."""
        C, H = fdict.get("C", 0), fdict.get("H", 0)
        N, O = fdict.get("N", 0), fdict.get("O", 0)
        P, S, Si = fdict.get("P", 0), fdict.get("S", 0), fdict.get("Si", 0)

        rules = {}
        rules["nitrogen_rule"] = (round(cls.compute_mass(fdict)) % 2 == N % 2)
        rules["valence_rule"] = (
            sum(cls.VALENCE.get(e, 0) * c for e, c in fdict.items()) % 2 == 0
        )
        rules["element_count"] = (
            C <= 100 and H <= 200 and N <= 20 and O <= 30 and P <= 6 and S <= 6
        )
        rules["H/C"] = cls.relaxed_hc_rule(C, H)

        if C > 0:
            if C <= 6:
                rules["N/C"] = (N <= 4)
            else:
                rules["N/C"] = (N / C <= 1.0 and N <= 14)
        else:
            rules["N/C"] = (N == 0)

        rules["O/C"] = (O <= C)
        rules["P/C"] = (P <= C / 3 + 1)
        rules["S/C"] = (S <= C / 2 + 1)
        rules["element_ratio_prob"] = (N <= 20 and O <= 30 and S <= 6 and P <= 6)
        rules["TMS_check"] = (Si == 0 or C > 0)
        return rules

    @staticmethod
    def element_accuracy(true_formula: str, pred_formula: str):
        """
        Strict element accuracy:
        fraction of elements with exactly correct counts.
        """
        true_dict = FormulaUtils.parse_formula_dict(true_formula)
        pred_dict = FormulaUtils.parse_formula_dict(pred_formula)
        element_list = sorted(set(true_dict.keys()) | set(pred_dict.keys()))
        correct, total = 0, 0
        for elem in element_list:
            if true_dict.get(elem, 0) == pred_dict.get(elem, 0):
                correct += 1
            total += 1
        return correct, total