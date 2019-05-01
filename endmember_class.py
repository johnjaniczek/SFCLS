import numpy as np
import pandas as pd
import copy
import spectral.io.envi as envi
from sklearn.preprocessing import LabelBinarizer
from glob import glob as gg
import csv

class spec_lib(object):
    def __init__(self, type, envi_hdr = "", envi_file = "", ascii_spectra = "", meta_csv = "", ascii_bands = "", directory_path = "" , meta_tab = ""):
        """
        loads a spectral library from common spectral library formats including crism (envi format), asu spectral library
        (ascii spectra and csv meta), and USGS (directory path and spectral bands ascii path)

        stores important spectral library features (spectra, spectral bands, name of spectra, and one hot labels) in a standard format

        Note: it is assumed that the first word of the "name" is the label of the mineral, which is generally true for all
        libraries with some exceptions. For these exceptions, the data should be relabelled.

        :param type: "asu", "crism", or "USGS"
        :param envi_hdr: path to envi header file (only for crism)
        :param envi_file: path to envi file (only for crism)
        :param ascii_spectra: path to ascii spectra file (ASU)
        :param meta_csv: path to meta csv file ( ASU)
        :param ascii_bands: path to spectral bands ascii file (USGS)
        :param director_path: path to directory (USGS)
        """

        #assign object variables per asu spec lib type
        if type == "asu":
            self.source = "asu"
            self.spectra = np.loadtxt(ascii_spectra)
            self.spectra = np.delete(self.spectra, 0, 1)  # delete first spectra column (wavenumbers)
            self.bands = np.loadtxt(ascii_spectra, usecols=0)
            self.meta = pd.read_csv(meta_csv)
            self.names = self.meta.sample_name.tolist()
            self.category = self.meta.category.tolist()

        if type == "kim":
            self.source = "kim"
            self.spectra = np.loadtxt(ascii_spectra)
            self.spectra = np.delete(self.spectra, 0, 1)  # delete first spectra column (wavenumbers)
            self.bands = np.loadtxt(ascii_spectra, usecols=0)
            with open(meta_tab) as f:
                self.names = list(csv.reader(f, delimiter='\t'))
            self.category = self.names
        
        # assign object variables per crism spec lib type
        if type == "crism":
            self.source = "crism"
            self.envi_file = envi.open(envi_hdr, envi_file)
            self.spectra = self.envi_file.spectra.transpose()
            self.bands = self.envi_file.bands.centers
            self.names = self.envi_file.names
            self.category = self.names

        if type == "usgs":
            self.source = "usgs"
            self.bands = np.loadtxt(ascii_bands, skiprows=1)
            #iterate through all txt files in directory path
            first = True
            for f in gg(directory_path):
                temp_spectra = np.loadtxt(f, skiprows=1)
                temp_spectra = temp_spectra.reshape((len(temp_spectra), 1))
                temp_meta = open(f, "r").readlines()[0].split()
                temp_name = temp_meta[2]

                if first:
                    self.spectra = temp_spectra
                    self.names = [temp_name]
                    first = False
                else:
                    self.spectra = np.append(self.spectra, temp_spectra, axis=1)
                    self.names.append(temp_name)
            self.text_labels = self.names
            self.category = self.names



        #assign general object variables
        self.text_labels = [names.partition(" ")[0] for names in self.names]
        self.index = range(len(self.names))
        self.src_index = range(len(self.names))
        encoder = LabelBinarizer()
        self.onehot_labels = encoder.fit_transform(self.text_labels)
        self.onehot_category = encoder.fit_transform(self.category)

    def subset(self, indices):
        """ returns a subset of the data given by the indices of the desired endmembers
        """
        sub = copy.deepcopy(self)
        sub.spectra = sub.spectra[:, indices]
        sub.names = [sub.names[i] for i in indices]
        sub.category = [sub.category[i] for i in indices]
        sub.text_labels = [sub.text_labels[i] for i in indices]
        sub.index = range(len(indices))
        sub.src_index = indices
        sub.onehot_labels = [sub.onehot_labels[i, :] for i in indices]
        sub.onehot_category= [sub.onehot_category[i, :] for i in indices]
        return sub

    def relabel(self):
        """
        redoes the onehot_labelling. use case is for a subset which now has fewer text labels and thus has some 0s for
        all onehot_label columns.
        """
        encoder = LabelBinarizer()
        self.onehot_labels = encoder.fit_transform(self.text_labels)
        self.onehot_category = encoder.fit_transform(self.category)
        
    def at_least(self, threshold):
        """
        returns a dataset with only endmembers that have 
        a count greater than the threshold
        :param threshold: minimum number of redundant samples
        :return: dataset with only endmember classes with counts greater than threshold
        """
        indices = []
        for i in range(self.onehot_labels.shape[1]):
            if sum(self.onehot_labels[:, i]) > threshold:
                for j in range(self.onehot_labels.shape[0]):
                    if self.onehot_labels[j, i]:
                        indices.append(j)
                        
        indices = sorted(indices)
        sub = self.subset(indices)
        sub.relabel()
        return sub

    def at_most(self, threshold):
        """
        returns a data set with no more redundant endmembers than the threshold
        :param threshold: limit of redundant endmembers
        :return: reduced set of endmembers
        """
        indices = []
        for i in range(self.onehot_labels.shape[1]):
            if sum(self.onehot_labels[:, i]) > threshold - 1:
                reduced_indices = np.nonzero(self.onehot_labels[:, i])[0]
                indices.extend([reduced_indices[j] for j in range(threshold)])

        indices = sorted(indices)
        sub = self.subset(indices)
        sub.relabel()
        return sub

    def augment(self, scale = 2, method="gauss", deviation = 1):
        aug = copy.deepcopy(self)
        if method == "gauss":
            for i in range(scale-1):
                for j in self.index:
                    #add gaussian noise to spectra
                    temp = self.spectra[:, j] + np.random.normal(0, deviation, self.spectra[:, j].shape)
                    
                    #append perturbed spectra and copy meta data
                    aug.spectra = np.append(aug.spectra, temp.reshape(len(temp), 1), axis=1)
                    aug.names.append(aug.names[j])
                    aug.category.append(aug.category[j])
                    aug.text_labels.append(aug.text_labels[j])
                    aug.onehot_labels = np.append(aug.onehot_labels,
                                                   aug.onehot_labels[j, :].reshape(1, aug.onehot_labels.shape[1]),
                                                   axis=0)
                    aug.onehot_category = np.append(aug.category_labels,
                                                    aug.category_labels[j, :].reshape(1, aug.onehot_category.shape[1]),
                                                    axis=0)
                    aug.src_index.append(self.src_index[j])
            
            aug.index = range(len(aug.names))
            
        return aug

    def perturb(self, method="rand_stretch", deviation=0.02, SNR=10, stretch=1.1, low=0.5, high=1.5):
        """
                perturbs a vector, mixture, with gaussian noise
                :param mixture: input vector
                :param method: method , currently only gauss is supported
                :param deviation: standard deviation of gauss method
                :return: mixtures_perturbed the perturbed mixture
        """
        endmemb_noisy = copy.deepcopy(self)

        if method == "gauss":
            # add gaussian noise
            noise = np.random.normal(0, deviation, endmemb_noisy.spectra.shape)
            endmemb_noisy.spectra = endmemb_noisy.spectra + noise

        if method == "lin_stretch":
            # stretch the variation of the signal linearly
            stretch_factor = stretch
            mean_spectra = endmemb_noisy.spectra.mean(axis=0)
            endmemb_noisy.spectra = stretch_factor * (endmemb_noisy.spectra - mean_spectra) + mean_spectra

        if method == "rand_stretch":
            # stretch the variation by a random factor
            stretch_factor = np.random.uniform(low=low, high=high, size=self.spectra.shape[1])
            mean_spectra = endmemb_noisy.spectra.mean(axis=0)
            endmemb_noisy.spectra = np.multiply(stretch_factor, (endmemb_noisy.spectra - mean_spectra)) + mean_spectra

        return endmemb_noisy




