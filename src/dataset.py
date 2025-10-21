import re
import numpy as np
import torch
from torch.utils.data import Dataset


class FormulaTokenizer:
    def __init__(self, elements=None, max_count=200):
        self.special_tokens = ["<pad>", "<sos>", "<eos>"]
        self.elements = elements or ["C", "H", "N", "O", "S", "P", "Cl", "Br", "F", "I"]
        # include multi-digit counts up to max_count
        self.counts = [str(i) for i in range(1, max_count + 1)]
        self.vocab = self.special_tokens + self.elements + self.counts
        self.stoi = {tok: i for i, tok in enumerate(self.vocab)}
        self.itos = {i: tok for tok, i in self.stoi.items()}
        self.max_count = max_count

    def encode(self, formula: str):
        tokens = ["<sos>"]
        for elem, count in re.findall(r'([A-Z][a-z]*)(\d*)', formula):
            if elem in self.stoi:
                tokens.append(elem)
            if count:
                if count in self.stoi:
                    tokens.append(count)
                else:
                    # clamp to max_count if larger
                    capped = str(min(int(count), self.max_count))
                    tokens.append(capped)
        tokens.append("<eos>")
        return [self.stoi[t] for t in tokens if t in self.stoi]

    def decode(self, ids):
        tokens = [self.itos[i] for i in ids if i in self.itos]
        formula = ""
        for tok in tokens:
            if tok in self.special_tokens:
                continue
            formula += tok
        return formula


def spectrum_to_vector(mz, intensities, max_mz=2000):
    vec = np.zeros(max_mz, dtype=np.float32)
    for m, inten in zip(mz, intensities):
        idx = int(round(m))
        if idx < max_mz:
            vec[idx] += inten
    if vec.sum() > 0:
        vec /= vec.sum()
    return vec


class SpectrumFormulaDataset(Dataset):
    def __init__(self, entries, tokenizer, max_mz=2000, max_len=30):
        self.entries = [e for e in entries if "Formula" in e]
        self.tokenizer = tokenizer
        self.max_mz = max_mz
        self.max_len = max_len

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        spectrum = spectrum_to_vector(entry['mz'], entry['intensities'], self.max_mz)
        tokens = self.tokenizer.encode(entry['Formula'])
        # pad sequence
        if len(tokens) < self.max_len:
            tokens = tokens + [self.tokenizer.stoi["<pad>"]] * (self.max_len - len(tokens))
        else:
            tokens = tokens[:self.max_len]
        spectrum = torch.tensor(spectrum, dtype=torch.float32)
        tokens = torch.tensor(tokens, dtype=torch.long)
        return spectrum, tokens