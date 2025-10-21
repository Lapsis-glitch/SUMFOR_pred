import re
from collections import defaultdict


def parse_formula(formula):
    """Parse a chemical formula like 'C17H37N7O3' into a dict."""
    pattern = r'([A-Z][a-z]*)(\d*)'
    elements = defaultdict(int)
    for elem, count in re.findall(pattern, formula):
        elements[elem] += int(count) if count else 1
    return dict(elements)

def parse_msp(file_path):
    entries = []
    entry = {}
    mzs, intensities = [], []

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                # end of entry
                if entry:
                    entry['mz'] = mzs
                    entry['intensities'] = intensities
                    if 'Formula' in entry:
                        entry['FormulaDict'] = parse_formula(entry['Formula'])
                    entries.append(entry)
                    entry, mzs, intensities = {}, [], []
                continue

            if ':' in line and not line.startswith('Num Peaks'):
                key, value = line.split(':', 1)
                entry[key.strip()] = value.strip()

            elif line.startswith('Num Peaks'):
                entry['Num Peaks'] = int(line.split(':', 1)[1].strip())

            else:
                # peak lines: may contain multiple pairs separated by ';'
                for token in line.split(';'):
                    token = token.strip()
                    if token:
                        parts = token.split()
                        if len(parts) == 2:
                            mz, intensity = parts
                            mzs.append(float(mz))
                            intensities.append(float(intensity))

    # catch last entry if file doesn’t end with blank line
    if entry:
        entry['mz'] = mzs
        entry['intensities'] = intensities
        if 'Formula' in entry:
            entry['FormulaDict'] = parse_formula(entry['Formula'])
        entries.append(entry)

    return entries

spectra = parse_msp("/home/rat/Leco/4Mix_comparison/4-Mix_Complete/NIST_mainlib/all.MSP")

print(f"Parsed {len(spectra)} entries")

first = spectra[40]
print("Name:", first.get("Name"))
print("FormulaDict:", first.get("FormulaDict"))
print("m/z array (first 10):", first['mz'][:10])
print("intensity array (first 10):", first['intensities'][:10])