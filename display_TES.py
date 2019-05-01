import pickle
import matplotlib.pyplot as plt
import cv2
from endmember_class import spec_lib
import numpy as np

with open("results/TES_unmix_raw/endmember_library_object.file", "rb") as f:
    endmembs = pickle.load(f)

mola_img = plt.imread("input/mola/mars_mola_resize360.png")

for method in ["SFCLS", "FCLS", "LASSO"]:
    with open("results/TES_unmix_raw/%s_abundance.file" %method, "rb") as f:
        abundance = pickle.load(f)
    with open("results/TES_unmix_raw/%s_RMS.file" %method, "rb") as f:
        RMS = pickle.load(f)

    for em in endmembs.index:
        plt.figure(em, figsize=(7.5, 3), dpi=int(200))
        min_map = plt.imshow(mola_img,)
        plt.imshow(cv2.resize(abundance[:, :, em],
                              (720, 360),
                              interpolation=cv2.INTER_NEAREST),
                   cmap="nipy_spectral",
                   alpha=0.4,)
        plt.title("%s %s" % (method, endmembs.names[em]))
        plt.colorbar()
        plt.axis("off")
        # plt.show()
        plt.savefig("results/mineral_maps/%s_%s.png" %(method, endmembs.names[em]))
        plt.close()

    plt.figure(figsize=(7.5, 3), dpi=int(200))
    min_map = plt.imshow(mola_img, )
    plt.imshow(cv2.resize(RMS,
                          (720, 360),
                          interpolation=cv2.INTER_NEAREST),
               cmap="nipy_spectral",
               alpha=0.4, )
    plt.title("%s RMS" % method)
    plt.colorbar()
    plt.axis("off")
    # plt.show()
    plt.savefig("results/mineral_maps/%s_RMS.png" % method)
    plt.close()
    print("%s RMS mean: %f" %(method, RMS.mean()))