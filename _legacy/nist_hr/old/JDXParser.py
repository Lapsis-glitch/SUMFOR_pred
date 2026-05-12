import re

def parse_jdx_mass(file_or_text):
    """
    Parse a JCAMP-DX mass spectrum file (NIST style).
    Returns a dict with:
      - metadata: dict of header fields
      - mz: list of nominal m/z values (ints)
      - intensity: list of intensities (ints)
    """
    # If a file path is given, read it
    if isinstance(file_or_text, str) and "\n" not in file_or_text:
        with open(file_or_text, "r") as f:
            lines = f.readlines()
    else:
        # Otherwise assume it's a text string
        lines = file_or_text.splitlines()

    metadata = {}
    mz = []
    intensity = []
    in_xydata = False

    for line in lines:
        line = line.strip()
        if not line:
            continue

        if line.startswith("##") and not in_xydata:
            if line.upper().startswith("##XYDATA"):
                in_xydata = True
                continue
            if "=" in line:
                key, val = line[2:].split("=", 1)
                metadata[key.strip()] = val.strip()
            else:
                metadata[line[2:].strip()] = None
        elif in_xydata:
            parts = line.split()
            if len(parts) >= 2:
                try:
                    mz_val = int(parts[0])
                    inten_val = int(parts[1])
                    mz.append(mz_val)
                    intensity.append(inten_val)
                except ValueError:
                    pass

    return {"metadata": metadata, "mz": mz, "intensity": intensity}