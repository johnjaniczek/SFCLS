"""
Experiment to predict abundances from laboratory mixtures
from the Feely dataset

inputs: input
"""


from mixclass import MixClass
from endmember_class import spec_lib
from unmix import FCLS_unmix, inftyNorm_unmix, LASSO_unmix, pNorm_unmix, deltaNorm_unmix
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import datetime

# experiment parameters
seed = 1
K = "N/A"
sigma = "N/A"
thresh = 0.01
run_FCLS = True
run_inftyNorm = True
run_LASSO = True
run_pnorm = True
run_delta = False

# algorithm hyperparameters
lam_LASSO = 1e-4
lam_infty = 1e-6
lam_delta = 1e-6
lam_pnorm = 1e-3
delta = 1e-2
p = 0.99

def main():
    # start from a random seed
    np.random.seed(seed)

    # initialize experiment
    endmembs = spec_lib("asu",
                        ascii_spectra="input/kim/kim_library_tab.txt",
                        meta_csv="input/kim/kim_library_meta.csv")

    mixtures = MixClass(ascii_spectra="input/kim/mtes_kimmurray_rocks_full_tab.txt",
                        meta_csv="input/kim/mtes_kimmurray_rocks_full_meta.csv")

    # crop spectra at 400 wavenumbers
    endmembs.spectra = endmembs.spectra[104:, :]
    mixtures.spectra = mixtures.spectra[104:, :]
    endmembs.bands = endmembs.bands[104:]

    metrics_FCLS = []
    metrics_inftyNorm = []
    metrics_LASSO = []
    metrics_pnorm = []
    metrics_delta = []

    # iterate over mixtures
    for i in range(len(mixtures.names)):
        # extract next mixture
        mixture = mixtures.single(i)
        idx_pos_truth = np.nonzero(mixture.presence > 0)[0]

        # check if mixture is labelled as valid
        if mixture.category == "valid_mixture":

            # spectral unmixing
            if run_inftyNorm:
                # SFCLS prediction
                x = inftyNorm_unmix(endmembs.spectra, mixture.spectra, lam=lam_infty)
                metrics_inftyNorm.append(Metrics(x, mixture, mixture, endmembs.spectra, idx_pos_truth))

            if run_FCLS:
                # FCLS prediction
                x = FCLS_unmix(endmembs.spectra, mixture.spectra)
                metrics_FCLS.append(Metrics(x, mixture, mixture, endmembs.spectra, idx_pos_truth))

            if run_LASSO:
                # LASSO prediction
                x = LASSO_unmix(endmembs.spectra, mixture.spectra, lam=lam_LASSO)
                x = x / sum(x)
                metrics_LASSO.append(Metrics(x, mixture, mixture, endmembs.spectra, idx_pos_truth))

            if run_pnorm:
                # p-norm prediction
                x = pNorm_unmix(endmembs.spectra, mixture.spectra, lam=lam_pnorm, p=p)
                metrics_pnorm.append(Metrics(x, mixture, mixture, endmembs.spectra, idx_pos_truth))

            if run_delta:
                # delta norm prediction
                x = deltaNorm_unmix(endmembs.spectra, mixture.spectra, lam=lam_delta, delta=delta)
                metrics_delta.append(Metrics(x, mixture, mixture, endmembs.spectra, idx_pos_truth))


    # save metrics
    result_path, today = create_directory()
    # create experiment index
    i = 0
    while os.path.exists(result_path + today + "_experiment_params%d.csv" % i):
        i += 1

    # store experiment parameters to export with results
    experiment_params = [{"seed": seed,
                          "K": K,
                          "sigma": sigma,
                          "thresh": thresh,
                          "run_FCLS": run_inftyNorm,
                          "run_LASSO": run_LASSO,
                          "run_pnorm": run_pnorm,
                          "run_delta": run_delta,
                          "lam_LASSO": lam_LASSO,
                          "lam_delta": lam_delta,
                          "lam_pnorm": lam_pnorm,
                          "delta": delta,
                          "p": p}]
    saveListOfDicts(experiment_params, "labmix_params", result_path, today, i)

    # save results of experiments
    if run_inftyNorm:
        saveListOfDicts(metrics_inftyNorm, "labmix_infty", result_path, today, i)

    if run_delta:
        saveListOfDicts(metrics_delta, "labmix_delta", result_path, today, i)

    if run_FCLS:
        saveListOfDicts(metrics_FCLS, "labmix_FCLS", result_path, today, i)

    if run_LASSO:
        saveListOfDicts(metrics_LASSO, "labmix_LASSO", result_path, today, i)

    if run_pnorm:
        saveListOfDicts(metrics_pnorm, "labmix_pnorm", result_path, today, i)



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

# Define metrics, computed for all algorithms using this function
def Metrics(x, mixture, mixture_noisy, em_spec, idx_pos_truth):
    N = len(x)
    recon = np.matmul(em_spec, x)
    x = consolidate(x)
    x_presence = np.zeros(x.shape)
    x_presence[np.nonzero(x >= thresh)] = 1
    precision = sum(mixture.presence[idx_pos_truth] == x_presence[idx_pos_truth]) / sum(x_presence)
    recall = sum(mixture.presence[idx_pos_truth] == x_presence[idx_pos_truth]) / sum(mixture.presence)
    metrics = {"RMS_true": ((1.0 / N) * sum((mixture.spectra - recon) ** 2)).item(),
               "Error_L1": (sum(abs((x - mixture.proportions.transpose())))).item(),
               "Error_L2": np.sqrt(sum((x - mixture.proportions.transpose()) ** 2)).item(),
               "accuracy": sum(mixture.presence == x_presence) / len(mixture.presence),
               "precision": precision,
               "recall": recall}
    return metrics



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

def saveListOfDicts(metrics, name, result_path, today, i):
    """
    :param metrics: list[dict1, dict2, ...]
    :param method: string, name of method
    :return:
    """
    metrics = pd.DataFrame(metrics)
    print(name)
    print(metrics.mean())
    metrics.to_csv(result_path + today + "_%s%s.csv" % (name, i))



if __name__ == '__main__':
    main()



