# This is a sample Python script.
import matplotlib.pyplot as plt
import numpy as np

from src.NISTMS import NISTMSConverter
from src.SpecRep import Repository
from src.utils import predict_formula_from_spectrum

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    pathNIST = "/home/rat/Leco/4Mix_comparison/4-Mix_Complete//NIST_Ref/"
    pathQCxMS = "/home/rat/Leco/4Mix_comparison/4-Mix_Complete/spectra_qcxms/"
    pathBTX = "/home/rat/Leco/4Mix_comparison/4-Mix_Complete/Benchtop/"
    pathAML = "/home/rat/Leco/4Mix_comparison/4-Mix_Complete/AML_renamed/"

    repoNIST = Repository(pathNIST, "/home/rat/Leco/4Mix_comparison/4-Mix_Complete/keys.csv")

    spec = repoNIST.get_by_id("1")
    print(spec.name)
    print(spec.mz[:5])
    print(spec.intensity[:5])

    converter = NISTMSConverter(rounding_mode='round', threshold=0.01, base_value=999)

    # repoQCxMS = Repository(pathNIST, "/home/rat/Leco/4Mix_comparison/4-Mix_Complete/keys.csv", spectraformat='jdx')
    repoQCxMS = Repository(pathQCxMS, "/home/rat/Leco/4Mix_comparison/4-Mix_Complete/keys.csv", spectraformat='jdx')

    repoQCxMS = Repository(pathBTX, "/home/rat/Leco/4Mix_comparison/4-Mix_Complete/keys.csv", spectraformat='jdx')

    print(repoNIST.list_ids())
    for sid in repoNIST.list_ids():
        if sid != "7":
            continue
        print("Found spectrum ID:", sid)
        spec = repoNIST.get_by_id(sid)
        # test BTX
        xrt = np.loadtxt("/home/rat/Leco/4Mix_comparison/4-Mix_Complete/Benchtop/Octane")

        xrt = np.loadtxt("/home/rat/Leco/4Mix_comparison/4-Mix_Complete/Benchtop/1-Octanol")# -> top fragment is C8H16, Water loss
        xrt = np.loadtxt("/home/rat/Leco/4Mix_comparison/4-Mix_Complete/Benchtop/2,6-Xylidine")
        print(xrt.shape)
        spec.mz = xrt[:,0]
        spec.intensity = xrt[:,1]
        converter = None

        plt.bar(spec.mz, spec.intensity, width=0.5)
        plt.xlabel("m/z")
        plt.ylabel("Intensity")
        plt.show()

        print(spec.mz, spec.intensity)

        candidates = predict_formula_from_spectrum(spec, converter, tol=0.5,rel_threshold=0.005, topN=15)#, precursor_override=144)
        if candidates:
            for i in range(len(candidates)):
                print(f"{spec.id} ({spec.name}): top candidate = {candidates[i][0]}  mass={candidates[i][1]:.4f}")
        else:
            print(f"{spec.id} ({spec.name}): no plausible candidates found")

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
