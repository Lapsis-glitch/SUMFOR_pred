import os
import csv
import numpy as np


class Spectrum:
    def __init__(self, id_, name, mz, intensity):
        self.id = id_
        self.name = name
        self.mz = np.array(mz, dtype=float)
        self.intensity = np.array(intensity, dtype=float)


class Repository:
    def __init__(self, folder, csv_path, spectraformat='jdx'):
        """
        Load spectra from folder and metadata from CSV.

        Parameters
        ----------
        folder : str
            Path to folder containing .jdx files.
        csv_path : str
            CSV file with two columns: ID, Name.
        """
        self.spectraformat = spectraformat
        self.folder = folder
        self.csv_path = csv_path
        self.spectra = {}  # id → Spectrum object
        # self.name_to_id = self._load_csv()
        self.id_to_name = self._load_csv()

        self._load_spectra()

    def _load_csv(self):

        mapping = {}
        with open(self.csv_path, 'r', encoding='utf-8') as f:
            # reader = csv.reader(f)
            for line in f:
                # print(line)
                id_, name = line.split()[0], line.split()[1]
                mapping[id_] = name
        return mapping

    # def _load_csv(self):
    #     mapping = {}
    #     with open(self.csv_path, 'r', encoding='utf-8') as f:
    #         reader = csv.reader(f)
    #         for row in reader:
    #             if len(row) >= 2:
    #                 id_, name = row[0].strip(), row[1].strip()
    #                 mapping[id_] = name
    #     return mapping

    def _parse_jdx(self, filepath):
        mz = []
        intensity = []
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        in_data = False
        for line in lines:
            line = line.strip()
            if line.startswith('##XYDATA=') or line.startswith('##PEAK TABLE'):
                in_data = True
                continue
            if line.startswith('##END='):
                break
            if in_data:
                parts = line.replace(',', ' ').split()
                if len(parts) == 2:
                    try:
                        mz_val = float(parts[0])
                        int_val = float(parts[1])
                        mz.append(mz_val)
                        intensity.append(int_val)
                    except ValueError:
                        continue
        return mz, intensity

    def _parse_csv(self, filepath):
        mz = []
        intensity = []
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        in_data = False
        for line in lines:
            if "Spectrum" in line:
                continue
            line = line.split(',')
            mz.append(line[2])
            intensity.append(line[4])
            # if line.startswith('##XYDATA='):
            #     in_data = True
            #     continue
            # if line.startswith('##END='):
            #     break
            # if in_data:
            #     parts = line.replace(',', ' ').split()
            #     if len(parts) == 2:
            #         try:
            #             mz_val = float(parts[0])
            #             int_val = float(parts[1])
            #             mz.append(mz_val)
            #             intensity.append(int_val)
            #         except ValueError:
            #             continue
        return mz, intensity

    # def _load_spectra(self):
    #     files = [f for f in os.listdir(self.folder) if f.lower().endswith('.jdx')]
    #     for filename in files:
    #         base = os.path.splitext(filename)[0]
    #         if base in self.name_to_id:
    #             id_ = self.name_to_id[base]
    #             name = base
    #             filepath = os.path.join(self.folder, filename)
    #             mz, intensity = self._parse_jdx(filepath)
    #             self.spectra[id_] = Spectrum(id_, name, mz, intensity)

    def _load_spectra(self):
        files = [f for f in os.listdir(self.folder) if f.lower().endswith('.jdx')]
        for filename in files:
            id_ = os.path.splitext(filename)[0]
            name = self.id_to_name.get(id_, "Unknown")
            filepath = os.path.join(self.folder, filename)
            if self.spectraformat == 'jdx':
                mz, intensity = self._parse_jdx(filepath)
            elif self.spectraformat == 'csv':
                mz, intensity = self._parse_csv(filepath)
            self.spectra[id_] = Spectrum(id_, name, mz, intensity)

    def get_by_id(self, id_):
        return self.spectra.get(id_)

    def get_by_name(self, name):
        id_ = self.name_to_id.get(name)
        return self.get_by_id(id_) if id_ else None

    def list_ids(self):
        return list(self.spectra.keys())

    def list_names(self):
        return list(self.name_to_id.keys())