from unmix import FCLS_unmix, LASSO_unmix, SFCLS_unmix, p_norm_unmix, delta_norm_unmix
import spectral.io.envi as envi
import numpy as np
from endmember_class import spec_lib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
import pickle
import os

# hyperparameters
ppd = 1     # pixels per degree
run_FCLS = True
run_SFCLS = False
run_LASSO = True
run_pnorm = True
run_delta = True
lam_LASSO = 1e-4
lam_SFCLS = 1e-6
lam_delta = 1e-6
lam_pnorm = 1e-3
delta = 1e-2
p = 0.99


# setup variables
file_path = "/home/john/datasets/TES/"
file_name = "TES_emissivity_4.csv"
step = 1/ppd
longitudes = np.arange(0, 360, step)
latitudes = np.arange(-90, 90, step)
Nx = 360*ppd
Ny = 180*ppd
idx_73pnt = np.arange(8, 35)
idx_73pnt = np.append(idx_73pnt, np.arange(64, 110))
channels = 73



# import dataframe of emissivities and metadata
print("loading data from:")
file_name = "TES_emissivity_0.csv"
df = pd.read_csv(file_path + file_name)
print(file_path + file_name)
for i in range(1, 7):
    file_name = "TES_emissivity_{}.csv".format(i)
    df = df.append(pd.read_csv(file_path + file_name))
    print(file_path + file_name)


# import bedrock endmembers
endmembs = spec_lib("asu_hdf",
                   hdf_file="/home/john/datasets/asu/bedrock_lib_v5_73pnts_highSi.hdf")
bedrock_idx = endmembs.index

# append atmospheric endmembers
atmosphere = spec_lib("asu_hdf",
                      hdf_file="/home/john/datasets/asu/atmlib_73channels.hdf")
endmembs.append(atmosphere)
Ne = endmembs.spectra.shape[1]

# setup output
FCLS_abundance = np.zeros((int(Nx), int(Ny), Ne))
FCLS_RMS = np.zeros((int(Nx), int(Ny)))
FCLS_SD = np.zeros((int(Nx), int(Ny), Ne))
SFCLS_abundance = np.zeros((int(Nx), int(Ny), Ne))
SFCLS_RMS = np.zeros((int(Nx), int(Ny)))
SFCLS_SD = np.zeros((int(Nx), int(Ny), Ne))
LASSO_abundance = np.zeros((int(Nx), int(Ny), Ne))
LASSO_RMS = np.zeros((int(Nx), int(Ny)))
LASSO_SD = np.zeros((int(Nx), int(Ny), Ne))
pnorm_abundance = np.zeros((int(Nx), int(Ny), Ne))
pnorm_RMS = np.zeros((int(Nx), int(Ny)))
pnorm_SD = np.zeros((int(Nx), int(Ny), Ne))
delta_abundance = np.zeros((int(Nx), int(Ny), Ne))
delta_RMS = np.zeros((int(Nx), int(Ny)))
delta_SD = np.zeros((int(Nx), int(Ny), Ne))
ObservationsPerPixel = np.zeros((int(Nx), int(Ny)))



# select spectra that belong in bins
# atmospherically correct and unmix spectra simultaneously
# but process individual spectra PRIOR to averaging the abundances for each bin
print("processing data")
for x, longitude in enumerate(longitudes):
    bin_long = (df["LONGITUDE_IAU2000"] < longitude + step) \
              & (df["LONGITUDE_IAU2000"] >= longitude)
    for y, latitude in enumerate(latitudes):
        bin_lat =(df["LATITUDE"] < latitude + step) \
                 & (df["LATITUDE"] >= latitude)
        if (bin_long & bin_lat).any():
            bin_members = df.loc[bin_long & bin_lat]
            emissivities = bin_members.iloc[:, 7:150]
            emissivities = emissivities.values[:, idx_73pnt]
            N_obs = emissivities.shape[0]
            ObservationsPerPixel[x, y] = N_obs

            FCLS_abund_temp = np.zeros((N_obs, Ne))
            FCLS_RMS_temp = np.zeros(N_obs)
            SFCLS_abund_temp = np.zeros((N_obs, Ne))
            SFCLS_RMS_temp = np.zeros(N_obs)
            LASSO_abund_temp = np.zeros((N_obs, Ne))
            LASSO_RMS_temp = np.zeros(N_obs)
            pnorm_abund_temp = np.zeros((N_obs, Ne))
            pnorm_RMS_temp = np.zeros(N_obs)
            delta_abund_temp = np.zeros((N_obs, Ne))
            delta_RMS_temp = np.zeros(N_obs)

            # process all spectra that fall in the latitude/longitude bin individually
            # then average the abundances by first summing the abundances from N observations and
            # then dividing by N, abund_bin = sum(abundance[n])/N_obs
            # similarly, average the RMS error for each bin such that bin_RMS = sum(RMS[n])/N_obs
            for obs, pixel in enumerate(emissivities):
                # if pixel is valid, then unmix
                if run_FCLS:
                    FCLS_abund_temp[obs] = FCLS_unmix(endmembs.spectra, pixel,
                                                      bedrock_idx=bedrock_idx)
                    recon = np.matmul(endmembs.spectra, FCLS_abund_temp[obs])
                    FCLS_RMS_temp[obs] += np.sqrt(sum((pixel - recon) ** 2) / len(pixel))
                if run_LASSO:
                    LASSO_abund_temp[obs] = LASSO_unmix(endmembs.spectra, pixel,
                                                 lam=lam_LASSO, bedrock_idx=bedrock_idx)
                    LASSO_abund_temp[obs] /= sum(LASSO_abund_temp[obs])
                    recon = np.matmul(endmembs.spectra, LASSO_abund_temp[obs])
                    LASSO_RMS_temp[obs] += np.sqrt(sum((pixel - recon) ** 2) / len(pixel))
                if run_SFCLS:
                    SFCLS_abund_temp[obs] = SFCLS_unmix(endmembs.spectra, pixel,
                                                 lam=lam_SFCLS, bedrock_idx=bedrock_idx)
                    recon = np.matmul(endmembs.spectra, SFCLS_abund_temp[obs])
                    SFCLS_RMS_temp[obs] += np.sqrt(sum((pixel - recon) ** 2) / len(pixel))
                if run_pnorm:
                    pnorm_abund_temp[obs] = p_norm_unmix(endmembs.spectra, pixel,
                                                 lam=lam_pnorm, p=p, bedrock_idx=bedrock_idx)
                    recon = np.matmul(endmembs.spectra, pnorm_abund_temp[obs])
                    delta_RMS_temp[obs] += np.sqrt(sum((pixel - recon) ** 2) / len(pixel))
                if run_delta:
                    delta_abund_temp[obs] = delta_norm_unmix(endmembs.spectra, pixel,
                                                 lam=lam_delta, delta=delta, bedrock_idx=bedrock_idx)
                    recon = np.matmul(endmembs.spectra, delta_abund_temp[obs])
                    delta_RMS_temp[obs] += np.sqrt(sum((pixel - recon) ** 2) / len(pixel))

            if run_FCLS:
                FCLS_abundance[x, y, :] = FCLS_abund_temp.mean(axis=0)
                FCLS_RMS[x, y] = FCLS_RMS_temp.mean()
                FCLS_SD[x, y, :] = FCLS_abund_temp.std(axis=0)
            if run_LASSO:
                LASSO_abundance[x, y, :] = FCLS_abund_temp.mean(axis=0)
                LASSO_RMS[x, y] = FCLS_RMS_temp.mean()
                LASSO_SD[x, y, :] = LASSO_abund_temp.std(axis=0)
            if run_SFCLS:
                SFCLS_abundance[x, y, :] = SFCLS_abund_temp.mean(axis=0)
                SFCLS_RMS[x, y] = SFCLS_RMS_temp.mean()
                SFCLS_SD[x, y, :] = SFCLS_abund_temp.std(axis=0)
            if run_pnorm:
                pnorm_abundance[x, y, :] = pnorm_abund_temp.mean(axis=0)
                pnorm_RMS[x, y] = pnorm_RMS_temp.mean()
                pnorm_SD[x, y, :] = pnorm_abund_temp.std(axis=0)
            if run_delta:
                delta_abundance[x, y, :] = delta_abund_temp.mean(axis=0)
                delta_RMS[x, y] = delta_RMS_temp.mean()
                delta_SD[x, y, :] = delta_abund_temp.std(axis=0)

        print("x:", x,"y:", y)

# save data with a pickle dump
# uncomment to iterate naming rather than overwrite
i = 0
# while os.path.exists("results/TES_unmix_raw/FCLS_abundance%s.file" % i):
#     i += 1

print("saving data")
# dump end member abundance and statistical metrics
if run_FCLS:
    with open("results/TES_unmix_raw/FCLS_abundance%s.file" % i, "wb") as f:
        pickle.dump(FCLS_abundance, f)
    with open("results/TES_unmix_raw/FCLS_RMS%s.file" % i, "wb") as f:
        pickle.dump(FCLS_RMS, f)
    with open("results/TES_unmix_raw/FCLS_SD%s.file" % i, "wb") as f:
        pickle.dump(FCLS_SD, f)

if run_LASSO:
    with open("results/TES_unmix_raw/LASSO_abundance%s.file" % i, "wb") as f:
        pickle.dump(LASSO_abundance, f)
    with open("results/TES_unmix_raw/LASSO_RMS%s.file" % i, "wb") as f:
        pickle.dump(LASSO_RMS, f)
    with open("results/TES_unmix_raw/LASSO_SD%s.file" % i, "wb") as f:
        pickle.dump(LASSO_SD, f)

if run_SFCLS:
    with open("results/TES_unmix_raw/SFCLS_abundance%s.file" % i, "wb") as f:
        pickle.dump(SFCLS_abundance, f)
    with open("results/TES_unmix_raw/SFCLS_RMS%s.file" % i, "wb") as f:
        pickle.dump(SFCLS_RMS, f)
    with open("results/TES_unmix_raw/SFCLS_SD%s.file" % i, "wb") as f:
        pickle.dump(SFCLS_SD, f)
if run_pnorm:
    with open("results/TES_unmix_raw/pnorm_abundance%s.file" % i, "wb") as f:
        pickle.dump(pnorm_abundance, f)
    with open("results/TES_unmix_raw/pnorm_RMS%s.file" % i, "wb") as f:
        pickle.dump(pnorm_RMS, f)
    with open("results/TES_unmix_raw/pnorm_SD%s.file" % i, "wb") as f:
        pickle.dump(pnorm_SD, f)
if run_delta:
    with open("results/TES_unmix_raw/delta_abundance%s.file" % i, "wb") as f:
        pickle.dump(delta_abundance, f)
    with open("results/TES_unmix_raw/delta_RMS%s.file" % i, "wb") as f:
        pickle.dump(delta_RMS, f)
    with open("results/TES_unmix_raw/delta_SD%s.file" % i, "wb") as f:
        pickle.dump(delta_SD, f)

# dump endmember library and observations per pixel data
with open("results/TES_unmix_raw/endmember_library_object%s.file" % i, "wb") as f:
    pickle.dump(endmembs, f)
with open("results/TES_unmix_raw/ObservationsPerPixel%s.file" % i, "wb") as f:
    pickle.dump(ObservationsPerPixel, f)

print("done")
