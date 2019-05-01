import numpy as np
import random
from mixclass import MixClass
from endmember_class import spec_lib

class SynthMix(object):
    def __init__(self, thresh=0.10):
        """
        Initialize the SynthMix class
        :param n_components: Number of components in the mixture
        """
        self.thresh=thresh

    def mix(self, A, n_components=2):
        """
        creates synthetic mixtures of input data A
        :param A: NxM matrix where each component of the mixture is stored as a column vector in A
        :return: mixture a synthetic mixture with N_components randomly chosen from A
        :return proportions, the true percentages of A in the mixture
        """

        N = A.shape[0]
        M = A.shape[1]

        # randomly select which components to use, and the percentage of each component
        components = random.sample(range(M), n_components)
        while True:
            percentages = np.random.uniform(low=self.thresh, high=1, size=n_components)
            percentages /= percentages.sum()
            if all(percentages >= self.thresh):
                break

        #create synthetic mixture and proportion vector
        mixture = MixClass()
        mixture.spectra = np.zeros(N)
        mixture.proportions = np.zeros(M)
        mixture.presence = np.zeros(M)
        for i in range(n_components):
            mixture.spectra += A[:, components[i]]*percentages[i]
            mixture.proportions[components[i]] = percentages[i]
            mixture.presence[components[i]] = 1
        return mixture

    def mars_mix(self, endmember_lib):
        """
        creates a synthetic mixture of endmember library A according to categorical distribution of mars
        :param endmember_lib: expecting an endmember class with martian minerals
        :return: synthetic mixture
        """
        N = endmember_lib.spectra.shape[0]
        M = endmember_lib.spectra.shape[1]
        components = []
        percentages = []

        # create basic mars distribution
        # select 1 Feldspar
        indices = [i for i, x in enumerate(endmember_lib.category) if x == "Feldspar (Plagioclase)"]
        components.extend(random.sample(indices, 1))
        percentages.append(random.uniform(0.1, 0.4))

        # select 2 pyroxenes
        indices = [i for i, x in enumerate(endmember_lib.category) if x == "Pyroxene"]
        components.extend(random.sample(indices, 2))
        percentages.append(random.uniform(0.05, 0.2))
        percentages.append(random.uniform(0.05, 0.2))

        # select 1 or 2 olivine
        indices = [i for i, x in enumerate(endmember_lib.category) if x == "Olivine"]
        n_comp = random.choice([1, 2])
        components.extend(random.sample(indices, n_comp))
        for i in range(n_comp):
            percentages.append(random.uniform(0, 0.2))

        # select 1 silicate
        indices = [i for i, x in enumerate(endmember_lib.category) if x == "Silicate"]
        components.extend(random.sample(indices, 1))
        percentages.append(random.uniform(0, 0.3))

        # select 1 carbonate
        indices = [i for i, x in enumerate(endmember_lib.category) if x == "Carbonate"]
        components.extend(random.sample(indices, 1))
        percentages.append(random.uniform(0, 0.2))

        # select 1 sulfate
        indices = [i for i, x in enumerate(endmember_lib.category) if x == "Sulfate"]
        components.extend(random.sample(indices, 1))
        percentages.append(random.uniform(0, 0.1))

        # choose 1 of 3 cases on mars randomly:
        # case 1: general mars distribution 90% of time
        # case 2: hematite rich mars distribution 5% of time
        # case 3: other random minerals 5% of time

        case = random.random()
        # case 1 normal mars distribution
        if case <= 0.9:
            percentages = [p / sum(percentages) for p in percentages]

        # case 2: hematite rich
        if 0.9 < case <= 0.95:
            indices = [i for i, x in enumerate(endmember_lib.category) if x == "Hematite"]
            components.extend(random.sample(indices, 1))
            percentages.append(random.uniform(0.01, 1))

            percentages[:-1] = [p * (1-percentages[-1]) / sum(percentages[:-1]) for p in percentages[:-1]]

        # case 3: add another random  minerals
        if 0.95 < case <= 1:
            indices = [i for i, x in enumerate(endmember_lib.category) if x == "Other"]
            components.extend(random.sample(indices, 1))
            percentages.append(random.uniform(0.01, 0.4))
            percentages[:-1] = [p * (1 - percentages[-1]) / sum(percentages[:-1]) for p in percentages[:-1]]

        # create synthetic mixture and proportion vector
        mixture = MixClass()
        mixture.spectra = np.zeros(N)
        mixture.proportions = np.zeros(M)
        mixture.presence = np.zeros(M)
        for i in range(len(components)):
            mixture.spectra += endmember_lib.spectra[:, components[i]] * percentages[i]
            mixture.proportions[components[i]] = percentages[i]
            if percentages[i] >= self.thresh:
                mixture.presence[components[i]] = 1
        return mixture












