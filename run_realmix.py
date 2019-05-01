"""
Experiment to predict abundances from laboratory mixtures
from the Feely dataset

inputs: input
"""


from mixclass import MixClass
from endmember_class import spec_lib
from unmix import FCLS_unmix, SFCLS_unmix, LASSO_unmix
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import datetime

# experiment parameters
SFCLS_lam = 1e-4
LASSO_lam = 1e-4
thresh = 0.01  # theshold to determine predicted endmember abundance presence


def consolidate(abundances):
    """
    consolidate endmember abundances into mineral abundances
    to compare ground truth (mineral category) with prediction (endmembers)
    """
    proportions = []
    # amphiboles
    proportions.append(abundances[0:3].sum())
    # consolidate biotite-chlorite
    idx = [3, 5, 23]
    proportions.append(abundances[idx].sum())
    # consolidate calcite-dolomite
    idx = [4, 6]
    proportions.append(abundances[idx].sum())
    # epidote
    proportions.append(abundances[9])
    # feldspar
    proportions.append(abundances[8:16].sum())
    # garnet, obsidian, glaucophane, kyanite, muscovite
    for j in [16, 17, 18, 19, 20]:
        proportions.append(abundances[j])
    # olivine
    proportions.append(abundances[21:23].sum())
    # pyroxene
    proportions.append(abundances[24:30].sum())
    # quartz
    proportions.append(abundances[30:32].sum())
    # serpentine
    proportions.append(abundances[[32, 37]].sum())
    #  talc, vesuvianite, zoisite
    for j in [33, 34, 35]:
        proportions.append(abundances[j])
    # normalize proportions by blackbody abundance
    proportions = [p / (1 - abundances[36]) for p in proportions]
    proportions = np.asarray(proportions)

    return proportions


def compute_metrics(x, spectra, mixture, lam):
    """
    :param x: predicted abundances
    :param spectra: endmember spectra
    :param mixture: mixclass object that is unmixed by x (ground truth)
    :param lam: lambda parameter used in unmixing, to store in results
    :return: recon, metrics
    """

    # consolidate endmembers into categories
    recon = np.matmul(spectra, x)
    x = consolidate(x)
    x_presence = np.zeros(x.shape)
    x_presence[np.nonzero(x >= thresh)] = 1

    # calculate loss metrics
    idx_pos_truth = np.nonzero(mixture.presence > 0)[0]
    precision = sum(mixture.presence[idx_pos_truth] == x_presence[idx_pos_truth]) / sum(x_presence)
    recall = sum(mixture.presence[idx_pos_truth] == x_presence[idx_pos_truth]) / sum(mixture.presence)
    metrics = {"lambda": lam,
               "RMS_true": np.sqrt(np.asscalar((1.0 / x.shape[0]) * sum((mixture.spectra - recon) ** 2))),
               "Error_L1": np.asscalar(sum(abs((x - mixture.proportions.transpose())))),
               "Error_L2": np.asscalar(sum((x - mixture.proportions.transpose()) ** 2)),
               "threshold": thresh,
               "accuracy": sum(mixture.presence == x_presence) / len(mixture.presence),
               "precision": precision,
               "recall": recall
               }

    return recon, metrics


def plot_reconstruction(x1, x2, x3,
                        recon1, recon2, recon3,
                        bands, mixture):
    """
    plot reconstructed signals
    :param x1: SFCLS abundances
    :param x2: FCLS abundances
    :param x3: LASSO abundances
    :param recon1: SFCLS reconstructed signal
    :param recon2: FCLS reconstructed signal
    :param recon3: LASSO reconstructed signal
    :param bands: wavenumbers for each xtick
    :param mixture: mixclass object (ground truth)
    :return: nothing
    """
    x1 = consolidate(x1)
    x2 = consolidate(x2)
    x3 = consolidate(x3)
    plt.figure(figsize=[11, 7.5])
    plt.subplot(211)
    plt.title("Signal Reconstruction")
    plt.plot(bands, mixture.spectra, label="original")
    plt.plot(bands, recon1, "--", label="SFCLS reconstructed")
    plt.plot(bands, recon2, "--", label="FCLS reconstructed")
    plt.plot(bands, recon3, "--", label="LASSO reconstructed")
    plt.xlabel("wavenumber")
    plt.ylabel("emissivity")
    plt.legend()

    # plot true mixture proportions and predicted proportions
    plt.subplot(212)
    plt.title("Endmember Proportions " + mixture.names)
    ind = np.arange(x1.shape[0])
    width = 0.35
    plt.bar(ind, mixture.proportions, width/3, label="true proportions")
    plt.bar(ind + width/3, x1, width/3, label="SFCLS prediction")
    plt.bar(ind + 2*width/3, x2, width / 3, label="FCLS prediction")
    plt.bar(ind + width, x3, width / 3, label="LASSO prediction")
    proportion_labels = ["amphiboles", "biotite-chlor",
                         "calcite-dol", "epidote",
                         "feldspar", "garnet", "obsidian",
                         "glaucophane", "kyanite", "muscovite",
                         "olivine", "pyroxene",
                         "quartz", "serpentine", "talc"
                                                 "vesuvianite", "zoisite"]
    plt.xticks(ind + width / 2, proportion_labels, rotation=90)

    plt.subplots_adjust(bottom=.16, hspace=.4)
    plt.legend()

def create_directory():
    now = datetime.datetime.now()
    # create directory for today's date
    today = now.strftime("%Y-%m-%d")

    todays_results = "results/" + today
    try:
        os.mkdir(todays_results)
    except OSError:
        print("Creation of the directory %s failed" % todays_results)
    else:
        print("Successfully created the directory %s " % todays_results)
    result_path = todays_results + "/"
    return result_path, today

def main():
    # initialize experiment
    result_path, today = create_directory()
    endmembs = spec_lib("asu",
                        ascii_spectra="input/kim/kim_library_tab.txt",
                        meta_csv="input/kim/kim_library_meta.csv")

    mixtures = MixClass(ascii_spectra="input/kim/mtes_kimmurray_rocks_full_tab.txt",
                        meta_csv="input/kim/mtes_kimmurray_rocks_full_meta.csv")

    # crop spectra at 400 wavenumbers
    endmembs.spectra = endmembs.spectra[104:, :]
    mixtures.spectra = mixtures.spectra[104:, :]
    endmembs.bands = endmembs.bands[104:]

    SFCLS_metrics = []
    FCLS_metrics = []
    LASSO_metrics = []

    # iterate over subset of mixtures
    for i in range(len(mixtures.names)):
        # create synthetic mixture
        mixture = mixtures.single(i)
        # mixture_noisy = mixture.perturb(method="gauss", deviation=gauss_dev)

        if mixture.category == "valid_mixture":
            # predict abundances
            x_SFCLS, loss = SFCLS_unmix(endmembs.spectra, mixture.spectra, lam=SFCLS_lam)
            x_FCLS, loss = FCLS_unmix(endmembs.spectra, mixture.spectra)
            x_LASSO, loss = LASSO_unmix(endmembs.spectra, mixture.spectra, lam=LASSO_lam)
            x_LASSO = x_LASSO / x_LASSO.sum()

            # compute metrics
            SFCLS_recon, metrics = compute_metrics(x_SFCLS, endmembs.spectra, mixture, SFCLS_lam)
            SFCLS_metrics.append(metrics)
            FCLS_recon, metrics = compute_metrics(x_FCLS, endmembs.spectra, mixture, "N/A")
            FCLS_metrics.append(metrics)
            LASSO_recon, metrics = compute_metrics(x_LASSO, endmembs.spectra, mixture, LASSO_lam)
            LASSO_metrics.append(metrics)

    #         # display comparison
    #         plot_reconstruction(x_SFCLS, x_FCLS, x_LASSO,
    #                             SFCLS_recon, FCLS_recon, LASSO_recon,
    #                             endmembs.bands, mixture)
    #
    # plt.show()

    # save metrics
    SFCLS_metrics = pd.DataFrame(SFCLS_metrics)
    FCLS_metrics = pd.DataFrame(FCLS_metrics)
    LASSO_metrics = pd.DataFrame(LASSO_metrics)
    print("SFCLS:")
    print(SFCLS_metrics.mean())
    print("FCLS:")
    print(FCLS_metrics.mean())
    print("LASSO:")
    print(LASSO_metrics.mean())
    i = 0
    while os.path.exists(result_path + today + "_SFCLS_realmix_metrics%s.csv" % i):
        i += 1
    SFCLS_metrics.to_csv(result_path + today + "_SFCLS_realmix_metrics%s.csv" % i)
    FCLS_metrics.to_csv(result_path + today + "_FCLS_realmix_metrics%s.csv" % i)
    LASSO_metrics.to_csv(result_path + today + "_LASSO_realmix_metrics%s.csv" % i)


if __name__ == '__main__':
    main()
