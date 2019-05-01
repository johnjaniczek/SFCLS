import numpy as np
import copy
import pandas as pd


class MixClass(object):
    def __init__(self, ascii_spectra=None, meta_csv=None):
        self.spectra = None
        self.proportions = None
        self.presence = None
        self.names = None
        self.category = None

        if ascii_spectra is not None:
            self.spectra = np.loadtxt(ascii_spectra)
            self.spectra = np.delete(self.spectra, 0, 1)  # delete first spectra column (wavenumbers)
            meta = pd.read_csv(meta_csv)
            self.names = meta.sample_name.tolist()
            self.category = meta.category.tolist()
            self.proportions = meta.iloc[:, 2:].values
            self.presence = np.zeros(self.proportions.shape)
            self.presence[np.where(self.proportions > 0)] = 1


    def single(self, index):
        sub = MixClass()
        sub.spectra = self.spectra[:, index]
        sub.names = self.names[index]
        sub.category = self.category[index]
        sub.proportions = self.proportions[index, :]
        sub.presence = self.presence[index, :]

        return sub

    def append(self, new_mixture):
        if self.spectra is not None:
            self.spectra = np.hstack((self.spectra, new_mixture.spectra))
        else:
            self.spectra = new_mixture.spectra

        if self.proportions is not None:
            self.proportions = np.vstack((self.proportions, new_mixture.proportions))
        else:
            self.proportions = new_mixture.proportions

        if self.presence is not None:
            self.presence = np.vstack((self.presence, new_mixture.presence))
        else:
            self.presence = new_mixture.presence

    def perturb(self, method="gauss", deviation=0.02, SNR=10, stretch=1.1):
        """
                perturbs a vector, mixture, with gaussian noise
                :param mixture: input vector
                :param method: method , currently only gauss is supported
                :param deviation: standard deviation of gauss method
                :return: mixtures_perturbed the perturbed mixture
        """
        mixture_noisy = copy.deepcopy(self)

        if method == "gauss":
            # add gaussian noise
            noise = np.random.normal(0, deviation, mixture_noisy.spectra.shape)
            mixture_noisy.spectra = mixture_noisy.spectra + noise

        if method == "lin_stretch":
            # stretch the variation of the signal linearly
            stretch_factor = stretch
            mean_spectra = mixture_noisy.spectra.mean(axis=1)
            mixture_noisy.spectra = stretch_factor * (mixture_noisy.spectra - mean_spectra) + mean_spectra

        if method == "rand_stretch":
            # stretch the variation by a random factor
            stretch_factor = np.random.uniform(low=0.5, high=1.5, size=self.spectra.shape[1])
            mean_spectra = mixture_noisy.spectra.mean(axis=1)
            mixture_noisy.spectra = np.multiply(stretch_factor, (mixture_noisy.spectra - mean_spectra)) + mean_spectra

        if method == "gauss_SNR":
            # add gaussian noise with a normalized SNR
            noise = np.random.normal(0, deviation, mixture_noisy.spectra.shape)
            a = np.sqrt((1.0 / SNR)*sum((mixture_noisy.spectra-mixture_noisy.spectra.max())**2) / sum(noise ** 2))
            noise = a*noise
            mixture_noisy.spectra = mixture_noisy.spectra + noise

        return mixture_noisy






