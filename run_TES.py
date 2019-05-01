from unmix import FCLS_unmix, LASSO_unmix, SFCLS_unmix
import spectral.io.envi as envi
import numpy as np
from endmember_class import spec_lib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
import os

#HYPER PARAMATERS
SFCLS_lam = 1e-5
LASSO_lam = 1e-5

# import image
img = envi.open('/home/john/datasets/TES/TES_surface_emissivity_1_ppd.envi.hdr',
                '/home/john/datasets/TES/TES_surface_emissivity_1_ppd.envi')
Nx, Ny, Nz = img.shape  # dimensions of img

# import end members
endmembs = spec_lib("asu",
                    ascii_spectra="/home/john/datasets/asu/rogers_tes73.txt",
                    meta_csv="/home/john/datasets/asu/rogers_meta.csv")
_, Ne = endmembs.spectra.shape

# solve for abundances (concentrations of each end member in each pixel)
FCLS_abundance = np.zeros((Nx, Ny, Ne))
LASSO_abundance = np.zeros((Nx, Ny, Ne))
SFCLS_abundance = np.zeros((Nx, Ny, Ne))
FCLS_RMS = np.zeros((Nx, Ny))
LASSO_RMS = np.zeros((Nx, Ny))
SFCLS_RMS = np.zeros((Nx, Ny))

for x in range(Nx):
    print(x,
          "SFCLS RMS mean: %f" % SFCLS_RMS.mean(),
          "FCLS RMS mean: %f" % FCLS_RMS.mean(),
          "LASSO RMS mean: %f" % LASSO_RMS.mean())
    for y in range(Ny):

        # if pixel is valid, then unmix
        pixel = img[x, y]
        if sum(pixel) > 0:
            FCLS_abundance[x, y, :], _ = FCLS_unmix(endmembs.spectra, pixel)
            LASSO_abundance[x, y, :], _ = LASSO_unmix(endmembs.spectra, pixel, lam=LASSO_lam)
            LASSO_abundance[x, y, :] = LASSO_abundance[x, y, :] / sum(LASSO_abundance[x, y, :])
            SFCLS_abundance[x, y, :], _ = SFCLS_unmix(endmembs.spectra, pixel, lam=SFCLS_lam)

            # compute performance metrics
            recon = np.matmul(endmembs.spectra, FCLS_abundance[x, y, :])
            FCLS_RMS[x, y] = np.sqrt(sum((pixel - recon) ** 2) / len(pixel))

            recon = np.matmul(endmembs.spectra, LASSO_abundance[x, y, :])
            LASSO_RMS[x, y] = np.sqrt(sum((pixel - recon) ** 2) / len(pixel))

            recon = np.matmul(endmembs.spectra, SFCLS_abundance[x, y, :])
            SFCLS_RMS[x, y] = np.sqrt(sum((pixel - recon) ** 2) / len(pixel))



i = 0
while os.path.exists("results/TES_unmix_raw/FCLS_abundance%s.file" % i):
    i += 1
# dump end member abudance for later usage
with open("results/TES_unmix_raw/FCLS_abundance%s.file" % i, "wb") as f:
    pickle.dump(FCLS_abundance, f)
with open("results/TES_unmix_raw/LASSO_abundance%s.file" % i, "wb") as f:
    pickle.dump(LASSO_abundance, f)
with open("results/TES_unmix_raw/SFCLS_abundance%s.file" % i, "wb") as f:
    pickle.dump(SFCLS_abundance, f)

# dump endmember library
with open("results/TES_unmix_raw/endmember_library_object%s.file" % i, "wb") as f:
    pickle.dump(endmembs, f)


# dump RMS error
with open("results/TES_unmix_raw/FCLS_RMS%s.file" % i, "wb") as f:
    pickle.dump(FCLS_RMS, f)
with open("results/TES_unmix_raw/LASSO_RMS%s.file" % i, "wb") as f:
    pickle.dump(LASSO_RMS, f)
with open("results/TES_unmix_raw/SFCLS_RMS%s.file" % i, "wb") as f:
    pickle.dump(SFCLS_RMS, f)


print("done")
