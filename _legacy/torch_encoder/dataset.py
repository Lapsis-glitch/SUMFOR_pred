import torch
from typing import List, Dict, Tuple

class MassSpectrumDataset(torch.utils.data.Dataset):
    """
    Dataset of spectra with molecular formula labels.
    """
    def __init__(self, spectra_list: List[List[Tuple[float, float]]],
                 formula_list: List[Dict[str, int]],
                 element_order: List[str],
                 max_peaks=500,
                 max_mz=2000.0):
        assert len(spectra_list) == len(formula_list)
        self.spectra = spectra_list
        self.formulas = formula_list
        self.element_order = element_order
        self.max_peaks = max_peaks
        self.max_mz = max_mz

    def __len__(self):
        return len(self.spectra)

    def __getitem__(self, idx):
        spectrum = self.spectra[idx]
        formula = self.formulas[idx]

        # Top-K by intensity, then sort by m/z
        spectrum = sorted(spectrum, key=lambda x: x[1], reverse=True)[:self.max_peaks]
        spectrum = sorted(spectrum, key=lambda x: x[0])

        mz = torch.tensor([p[0] / self.max_mz for p in spectrum], dtype=torch.float32)
        intens = torch.tensor([p[1] for p in spectrum], dtype=torch.float32)
        if intens.numel() > 0 and intens.max() > 0:
            intens = intens / intens.max()

        # Pad
        T = len(spectrum)
        if T < self.max_peaks:
            mz = torch.cat([mz, torch.zeros(self.max_peaks - T)])
            intens = torch.cat([intens, torch.zeros(self.max_peaks - T)])

        mask = torch.zeros(self.max_peaks, dtype=torch.bool)
        mask[:T] = True

        counts = torch.tensor([formula.get(e, 0) for e in self.element_order], dtype=torch.long)

        return mz, intens, mask, counts


def collate_batch(batch):
    """
    Collate variable spectra into tensors for DataLoader.
    Args:
        batch: list of tuples (mz, intens, mask, counts)
    Returns:
        mz: (B, T)
        intens: (B, T)
        mask: (B, T)
        counts: (B, E)
    """
    mz, intens, mask, counts = zip(*batch)
    mz = torch.stack(mz, dim=0)
    intens = torch.stack(intens, dim=0)
    mask = torch.stack(mask, dim=0)
    counts = torch.stack(counts, dim=0)
    return mz, intens, mask, counts