from mixclass import MixClass
from synthmix import SynthMix
from endmember_class import spec_lib
from unmix import gravitron
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import copy
import os
import datetime
now = datetime.datetime.now()

# experiment parameters
infty_lam = 0                         # weight of p_norm regularization
p_lam = 1e-6
delta_lam = 0
K = 100                               # number of mixtures to test
gauss_dev = 2.5e-4                     # standard deviation of gaussian noise added to mixture
min_endmemb = 0.01                   # threshold to determine true endmember abundance present
thresh = 0.01                       # theshold to determine predicted endmember abundance presence
p = .8
delta = 1e-2


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

# initialize experiment
np.random.seed(1)
random.seed(1)
endmembs = spec_lib("asu",
                    ascii_spectra="input/endmemb_libs/rogers_tes73.txt",
                    meta_csv="input/endmemb_libs/rogers_meta.csv")
synth = SynthMix(thresh=min_endmemb)
metrics = []
N = endmembs.spectra.shape[0]           # number of spectral channels
M = endmembs.spectra.shape[1]           # number of endmembers

# iterate over K mixtures
for i in range(K):
    # create synthetic mixture
    mixture = synth.mars_mix(endmembs)
    mixture_noisy = mixture.perturb(method="gauss", deviation=gauss_dev)

    # predict abundances and presence
    x, loss = gravitron(endmembs.spectra, mixture_noisy.spectra, infty_lam=infty_lam,
                        p_lam=p_lam, delta_lam=delta_lam, delta=delta, p=p)
    x_presence = np.zeros(x.shape)
    x_presence[np.nonzero(x >= thresh)] = 1

    # calculate loss metrics
    idx_pos_truth = np.nonzero(mixture.presence > 0)[0]
    idx_neg_truth = np.nonzero(mixture.presence == 0)[0]
    precision = sum(mixture.presence[idx_pos_truth] == x_presence[idx_pos_truth]) / sum(x_presence)
    recall = sum(mixture.presence[idx_pos_truth] == x_presence[idx_pos_truth]) / sum(mixture.presence)
    metrics.append({"infty_lambda": infty_lam,
                    "p_lambda": infty_lam,
                    "delta_lam": delta_lam,
                    "delta": delta,
                    "p": p,
                    "SNR": np.asscalar(sum((mixture.spectra - mixture.spectra.max())**2) / sum((mixture.spectra - mixture_noisy.spectra)**2)),
                    "RMS_noisy": np.asscalar((1.0/N)*sum((mixture_noisy.spectra - np.matmul(endmembs.spectra, x))**2)),
                    "RMS_true": np.asscalar((1.0/N)*sum((mixture.spectra - np.matmul(endmembs.spectra, x))**2)),
                    "Error_L1": np.asscalar(sum(abs((x - mixture.proportions.transpose())))),
                    "Error_L2": np.asscalar(sum((x - mixture.proportions.transpose())**2)),
                    "threshold": thresh,
                    "accuracy": sum(mixture.presence == x_presence) / len(mixture.presence),
                    "precision": precision,
                    "recall": recall,
                    "K": K})

#     # plot true spectra, noisy spectra, and reconstructed spectra
#     plt.figure(figsize=[11, 7.5])
#     plt.subplot(211)
#     plt.title("Signal Reconstruction")
#     original = np.ma.array(mixture.spectra)
#     noisy = np.ma.array(mixture_noisy.spectra)
#     recon = np.ma.array(np.matmul(endmembs.spectra, x))
#     original[26] = np.ma.masked
#     noisy[26] = np.ma.masked
#     recon[26] = np.ma.masked
#     plt.plot(endmembs.bands, original, label="original")
#     plt.plot(endmembs.bands, noisy, label="noisy")
#     plt.plot(endmembs.bands, recon, "--", label="reconstructed")
#     plt.xlabel("wavenumber")
#     plt.ylabel("emissivity")
#     plt.legend()
#
#     # plot true mixture proportions and predicted proportions
#     plt.subplot(212)
#     plt.title("Endmember Proportions")
#     ind = np.arange(M)
#     width = 0.35
#     plt.bar(ind, mixture.proportions, width, label="true proportions")
#     plt.bar(ind + width, x, width, label="prediction")
#     plt.xticks(ind + width / 2, endmembs.text_labels, rotation=90)
#
#     plt.subplots_adjust(bottom=.16, hspace=.4)
#     plt.legend()
# plt.show()


#save metrics
metrics = pd.DataFrame(metrics)
print(metrics.mean())
i = 0
while os.path.exists(result_path + today + "_gravitron_metrics%s.csv" % i):
    i += 1
metrics.to_csv(result_path + today + "_gravitron_metrics%s.csv" % i)






