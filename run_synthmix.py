from mixclass import MixClass
from synthmix import SynthMix
from endmember_class import spec_lib
from unmix import inftyNorm_unmix, FCLS_unmix, LASSO_unmix, pNorm_unmix
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import copy
import os
import datetime
now = datetime.datetime.now()

# experiment parameters
seed = 1
K = 1000
sigma = 2.5e-4
min_endmemb = 0.01
thresh = 0.01
run_FCLS = True
run_infty = True
run_LASSO = True
run_pnorm = True

# algorithm hyperparameters
lam_LASSO = 1e-6
lam_infty = 3e-6
lam_pnorm = 1e-2
p = 0.999

# start from a random seed
np.random.seed(seed)
random.seed(seed)


# Define metrics, computed for all algorithms using this function
def Metrics(x, mixture, mixture_noisy, em_spec, idx_pos_truth):
    x_presence = np.zeros(x.shape)
    x_presence[np.nonzero(x >= thresh)] = 1
    precision = sum(mixture.presence[idx_pos_truth] == x_presence[idx_pos_truth]) / sum(x_presence)
    recall = sum(mixture.presence[idx_pos_truth] == x_presence[idx_pos_truth]) / sum(mixture.presence)
    metrics = {"RMS_noisy": ((1.0 / N) * sum((mixture_noisy.spectra - em_spec @ x) ** 2)).item(),
               "RMS_true": ((1.0 / N) * sum((mixture.spectra - em_spec@x) ** 2)).item(),
               "Error_L1": (sum(abs((x - mixture.proportions.transpose())))).item(),
               "Error_L2": np.sqrt(sum((x - mixture.proportions.transpose()) ** 2)).item(),
               "accuracy": sum(mixture.presence == x_presence) / len(mixture.presence),
               "precision": precision,
               "recall": recall}
    return metrics

# save metrics
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


#import endmembers
endmembs = spec_lib("asu",
                    ascii_spectra="input/endmemb_libs/rogers_tes73.txt",
                    meta_csv="input/endmemb_libs/rogers_meta.csv")

# initialize some objects
synth = SynthMix(thresh=min_endmemb)
metrics_FCLS = []
metrics_inftyNorm = []
metrics_LASSO = []
metrics_pnorm = []
N = endmembs.spectra.shape[0]           # number of spectral channels
M = endmembs.spectra.shape[1]           # number of endmembers

# iterate over K mixtures
for i in range(K):
    # create synthetic mixture
    mixture = synth.mars_mix(endmembs)
    idx_pos_truth = np.nonzero(mixture.presence > 0)[0]
    idx_neg_truth = np.nonzero(mixture.presence == 0)[0]
    mixture_noisy = mixture.perturb(method="gauss", deviation=sigma)

    if run_infty:
        # SFCLS prediction
        x = inftyNorm_unmix(endmembs.spectra, mixture_noisy.spectra, lam=lam_infty)
        metrics_inftyNorm.append(Metrics(x, mixture, mixture_noisy, endmembs.spectra, idx_pos_truth))

    if run_FCLS:
        # FCLS prediction
        x = FCLS_unmix(endmembs.spectra, mixture_noisy.spectra)
        metrics_FCLS.append(Metrics(x, mixture, mixture_noisy, endmembs.spectra, idx_pos_truth))

    if run_LASSO:
        # LASSO prediction
        x = LASSO_unmix(endmembs.spectra, mixture_noisy.spectra, lam=lam_LASSO)
        x = x/sum(x)
        metrics_LASSO.append(Metrics(x, mixture, mixture_noisy, endmembs.spectra, idx_pos_truth))

    if run_pnorm:
        # p-norm prediction
        x = pNorm_unmix(endmembs.spectra, mixture_noisy.spectra, lam=lam_pnorm, p=p)
        metrics_pnorm.append(Metrics(x, mixture, mixture_noisy, endmembs.spectra, idx_pos_truth))

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

# create experiment index
i = 0
while os.path.exists(result_path + today + "_params%d.csv" % i):
    i += 1


# store experiment parameters to export with results
experiment_params = [{"seed": seed,
                     "K": K,
                     "sigma": sigma,
                     "min_endmemb": min_endmemb,
                     "thresh": thresh,
                     "run_FCLS": run_infty,
                     "run_LASSO": run_LASSO,
                     "run_pnorm": run_pnorm,
                     "lam_LASSO": lam_LASSO,
                     "lam_pnorm": lam_pnorm,
                     "p": p}]
saveListOfDicts(experiment_params, "synthmix_params", result_path, today, i)

# save results of experiments
if run_infty:
    saveListOfDicts(metrics_inftyNorm, "synthmix_infty", result_path, today, i)

if run_FCLS:
    saveListOfDicts(metrics_FCLS, "synthmix_FCLS", result_path, today, i)

if run_LASSO:
    saveListOfDicts(metrics_LASSO, "synthmix_LASSO", result_path, today, i)

if run_pnorm:
    saveListOfDicts(metrics_pnorm, "synthmix_pnorm", result_path, today, i)







