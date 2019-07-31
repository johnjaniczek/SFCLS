from synthmix import SynthMix
from endmember_class import spec_lib
from unmix import SFCLS_unmix, FCLS_unmix, LASSO_unmix, p_norm_unmix, delta_norm
import numpy as np
import pandas as pd
import copy
import matplotlib.pyplot as plt
import random
import os
import datetime
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


# experiment parameters
SFCLS_lam = 1e-6    # weight of SFCLS regularization
LASSO_lam = 1e-3    # weight of LASSO regularization
p_norm_lam = 1e-5    # weight of p-norm regularization
delta_lam = 1e-6    # weight of delta norm regularization
p = 0.8        # order of p-norm
delta = 1e-2        # scale of delta norm
sigmas = np.logspace(-6, -3, 31)

K = 50                        # number of mixtures to test per sweep
min_endmemb = 0.01              # minimum endmember percentage

# initialize metrics
SFCLS_Error_L1 = []             # average L1 error on abundance prediction
FCLS_Error_L1 = []
LASSO_Error_L1 = []
p_norm_Error_L1 = []
delta_Error_L1 = []

results = pd.DataFrame()
# sweep over the parameter lambda
for sigma in sigmas:

    # reset total error counts for mean error metric
    SFCLS_Error_Total = 0
    FCLS_Error_Total = 0
    LASSO_Error_Total = 0
    p_norm_Error_Total = 0
    delta_Error_Total = 0

    # initialize experiment
    np.random.seed(1)
    random.seed(1)
    endmembs = spec_lib("asu",
                        ascii_spectra="input/endmemb_libs/rogers_tes73.txt",
                        meta_csv="input/endmemb_libs/rogers_meta.csv")
    synth = SynthMix(thresh=min_endmemb)
    metrics = []
    N = endmembs.spectra.shape[0]  # number of spectral channels
    M = endmembs.spectra.shape[1]  # number of endmembers

    # iterate over K mixtures
    for i in range(K):
        # create synthetic mixture
        mixture = synth.mars_mix(endmembs)
        mixture_noisy = mixture.perturb(method="gauss", deviation=sigma)

        # SFCLS prediction
        x, _ = SFCLS_unmix(endmembs.spectra, mixture_noisy.spectra, lam=SFCLS_lam)
        SFCLS_Error_Total += np.asscalar(sum(abs(x - mixture.proportions.transpose())))

        # FCLS prediction
        x, _ = FCLS_unmix(endmembs.spectra, mixture_noisy.spectra)
        FCLS_Error_Total += np.asscalar(sum(abs(x - mixture.proportions.transpose())))

        # LASSO prediction
        x, _ = LASSO_unmix(endmembs.spectra, mixture_noisy.spectra, lam=LASSO_lam)
        LASSO_Error_Total += np.asscalar(sum(abs(x - mixture.proportions.transpose())))

        # p-norm prediction
        x, _ = p_norm_unmix(endmembs.spectra, mixture_noisy.spectra, lam=p_norm_lam, p=p)
        p_norm_Error_Total += np.asscalar(sum(abs(x - mixture.proportions.transpose())))

        # delta norm prediction
        x, _ = delta_norm(endmembs.spectra, mixture_noisy.spectra, lam=delta_lam, delta=delta)
        delta_Error_Total += np.asscalar(sum(abs(x - mixture.proportions.transpose())))

    # store average metrics
    SFCLS_Error_L1.append((1.0/K) * SFCLS_Error_Total)
    FCLS_Error_L1.append((1.0/K) * FCLS_Error_Total)
    LASSO_Error_L1.append((1.0/K) * LASSO_Error_Total)
    p_norm_Error_L1.append((1.0 / K) * p_norm_Error_Total)
    delta_Error_L1.append((1.0/K) * delta_Error_Total)

    print("sigma:", sigma)


# store results
results = pd.DataFrame({"SFCLS_err_L1": SFCLS_Error_L1,
                        "FCLS_err_L1": FCLS_Error_L1,
                        "LASSO_err_L1": LASSO_Error_L1,
                        "p_norm_err_L1": p_norm_Error_L1,
                        "delta_err_L1": delta_Error_L1,
                        "SFCLS_lam": SFCLS_lam,
                        "LASSO_lam": LASSO_lam,
                        "p_norm_lam": p_norm_lam,
                        "delta_lam": delta_lam,
                        "p": p,
                        "delta": delta,
                        "Deviation": sigmas})

i = 0
while os.path.exists(result_path + today + "_noise_sweep_results%s.csv" % i):
    i += 1
results.to_csv(result_path + today + "_noise_sweep_results%s.csv" % i)

