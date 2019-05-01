import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# load FCLS results
path = "results/2019-03-25/"
filename = "2019-03-25_FCLS_metrics1"
FCLS_results = pd.read_csv(path + filename + ".csv")

# load LASSO results
path = "results/2019-03-25/"
filename = "2019-03-25_LASSO_metrics1"
LASSO_results = pd.read_csv(path + filename + ".csv")

# load SFCLS results
path = "results/2019-03-25/"
filename = "2019-03-25_SFCLS_metrics1"
SFCLS_results = pd.read_csv(path + filename + ".csv")

SFCLS_metrics = [SFCLS_results["accuracy"].mean(),
                 SFCLS_results["precision"].mean(),
                 SFCLS_results["recall"].mean(),
                 SFCLS_results["Error_L1"].mean(),
                 SFCLS_results["RMS_noisy"].mean(),
                 SFCLS_results["RMS_true"].mean()]

FCLS_metrics = [FCLS_results["accuracy"].mean(),
                FCLS_results["precision"].mean(),
                FCLS_results["recall"].mean(),
                FCLS_results["Error_L1"].mean(),
                FCLS_results["RMS_noisy"].mean(),
                FCLS_results["RMS_true"].mean()]

LASSO_metrics = [LASSO_results["accuracy"].mean(),
                 LASSO_results["precision"].mean(),
                 LASSO_results["recall"].mean(),
                 LASSO_results["Error_L1"].mean(),
                 LASSO_results["RMS_noisy"].mean(),
                 LASSO_results["RMS_true"].mean()]

metrics = ["accuracy", "precision", "recall", "error", "RMS_noisy", "RMS_true"]

# create dataframe comparison
comparison_df = pd.DataFrame({"SFCLS": SFCLS_metrics,
                              "FCLS": FCLS_metrics,
                              "LASSO": LASSO_metrics},
                             index=metrics)
print(comparison_df)
i = 0
while os.path.exists(path + "comparison%s.csv" % i):
    i += 1
comparison_df.to_csv(path + "comparison%s.csv" % i)

# display bar graph comparison
plt.figure()
plt.title("Spectral Unmixing Algorithm Comparison")
width = 0.2
ind = np.arange(4)
plt.bar(ind, SFCLS_metrics[0:4], width, label="SFCLS")
plt.bar(ind + width, FCLS_metrics[0:4], width, label="FCLS")
plt.bar(ind + 2 * width, LASSO_metrics[0:4], width, label="LASSO")
plt.xticks(ind + width / 3, metrics)
plt.legend()
plt.show()