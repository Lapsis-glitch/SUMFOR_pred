import numpy as np
import re

def parse_jdx(filepath):
    """
    Parse a JCAMP-DX (.jdx) file and extract m/z and intensity arrays.

    Parameters
    ----------
    filepath : str
        Path to the .jdx file.

    Returns
    -------
    mz : np.ndarray
        Array of m/z values (float).
    intensity : np.ndarray
        Array of intensity values (float).
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    mz = []
    intensity = []
    in_data = False

    for line in lines:
        line = line.strip()

        # Start of data block
        # print(line)
        if line.startswith('##PEAK TABLE=(XY..XY) 1'):
            in_data = True
            continue

        # End of data block
        if line.startswith('##END='):
            break

        if in_data:
            # Handle compressed or delimited XY data
            # Format: "m/z intensity" or "m/z,intensity"
            parts = re.split(r'[,\s]+', line)
            if len(parts) == 2:
                try:
                    mz_val = float(parts[0])
                    int_val = float(parts[1])
                    mz.append(mz_val)
                    intensity.append(int_val)
                except ValueError:
                    continue  # skip malformed lines

    if not mz or not intensity:
        raise ValueError("No valid spectral data found in file.")

    return np.array(mz), np.array(intensity)