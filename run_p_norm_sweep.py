from synthmix import SynthMix
from endmember_class import spec_lib
from unmix import p_norm_unmix
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
p_range = np.linspace(0.5, 1, 51)
lambdas = [1e-7]                # weights of L1 regularization term
K = 50                          # number of mixtures to test
gauss_dev = 2.5e-5              # standard deviation of gaussian noise added to mixture
min_endmemb = 0.01              # minimum endmember percentage
thresh = 0.01                   # threshold for determining presence of endmembers

# initialize metrics
Error_L1_mean = []             # average L1 error on abundance prediction
Error_L2_mean = []             # average L2 error on abundance prediction
RMS_noisy_mean = []            # RMS error on signal reconstruction - noisy signal averaged over K mixtures
RMS_true_mean = []             # RMS error on signal reconstruction - true signal averaged over K mixtures
accuracy_mean = []
precision_mean = []
recall_mean = []
SNR_mean = []                  # SNR averaged over K mixtures
p_list = []
lam_list = []

results = pd.DataFrame()

# sweep over the parameters p and lambda
for p in p_range:
    for lam in lambdas:
        RMS_noisy_total = 0
        RMS_true_total = 0
        Error_L1_total = 0
        Error_L2_total = 0
        SNR_total = 0
        true_positive_total = 0
        true_negative_total = 0
        accuracy_total = 0
        precision_total = 0
        recall_total = 0

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
            mixture_noisy = mixture.perturb(method="gauss", deviation=gauss_dev)

            # predict abundances and presence
            x, loss = p_norm_unmix(endmembs.spectra, mixture_noisy.spectra, lam=lam, p=p)
            x_presence = np.zeros(x.shape)
            x_presence[np.nonzero(x > thresh)] = 1

            # calculate loss metrics
            idx_pos_truth = np.nonzero(mixture.presence > 0)[0]
            idx_neg_truth = np.nonzero(mixture.presence == 0)[0]
            RMS_noisy_total += np.asscalar((1.0 / N) * sum((mixture_noisy.spectra - np.matmul(endmembs.spectra, x)) ** 2))
            RMS_true_total += np.asscalar((1.0 / N) * sum((mixture.spectra - np.matmul(endmembs.spectra, x)) ** 2))
            Error_L1_total += np.asscalar(sum(abs(x - mixture.proportions.transpose())))
            Error_L2_total += np.asscalar(sum((x - mixture.proportions.transpose()) ** 2))
            accuracy_total += sum(mixture.presence == x_presence) / len(mixture.presence)
            precision_total += sum(mixture.presence[idx_pos_truth] == x_presence[idx_pos_truth]) / sum(x_presence)
            recall_total += sum(mixture.presence[idx_pos_truth] == x_presence[idx_pos_truth]) / sum(mixture.presence)
            SNR_total += np.asscalar(sum((mixture.spectra - mixture.spectra.max())**2) / sum((mixture.spectra - mixture_noisy.spectra)**2))

        # store average metrics
        Error_L1_mean.append((1.0/K) * Error_L1_total)
        Error_L2_mean.append((1.0/K) * Error_L2_total)
        accuracy_mean.append((1.0/K) * accuracy_total)
        precision_mean.append((1.0/K) * precision_total)
        recall_mean.append((1.0/K) * recall_total)
        RMS_noisy_mean.append((1.0/K) * RMS_noisy_total)
        RMS_true_mean.append((1.0/K) * RMS_true_total)
        SNR_mean.append((1.0/K) * SNR_total)
        p_list.append(p)
        lam_list.append(lam)

        print("p: %f, lambda: %f, Error_L1_mean: %f" % (p, lam, Error_L1_mean[-1]))


# store results
results = pd.DataFrame({"lambda": lam_list,
                        "thresh": thresh,
                        "K": K,
                        "SNR": SNR_mean,
                        "Error_L1_mean": Error_L1_mean,
                        "Error_L2_mean": Error_L2_mean,
                        "accuracy_mean": accuracy_mean,
                        "precision_mean": precision_mean,
                        "recall_mean": recall_mean,
                        "RMS_noisy_mean": RMS_noisy_mean,
                        "RMS_true_mean": RMS_true_mean,
                        "Deviation": gauss_dev,
                        "p": p_list})

i = 0
while os.path.exists(result_path + today + "_p_norm_sweep_results%s.csv" % i):
    i += 1
results.to_csv(result_path + today + "_p_norm_sweep_results%s.csv" % i)

