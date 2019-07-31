import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# comparison parameters
FCLS = True
LASSO = True
inftyNorm = True
pnorm = True
delta = False

date = "2019-07-11"
run_num = "0"
experiment = "labmix"
path = "results/" + date + "/"




# load FCLS results
if FCLS:
    filename = date + "_" + experiment + "_FCLS" + run_num + ".csv"
    FCLS_results = pd.read_csv(path + filename)

# load LASSO results
if LASSO:
    filename = date + "_" + experiment + "_LASSO" + run_num + ".csv"
    LASSO_results = pd.read_csv(path + filename)

# load inftyNorm results
if inftyNorm:
    filename = date + "_" + experiment + "_infty" + run_num + ".csv"
    inftyNorm_results = pd.read_csv(path + filename)

if pnorm:
    filename = date + "_" + experiment + "_pnorm" + run_num + ".csv"
    pnorm_results = pd.read_csv(path + filename)

inftyNorm_metrics = [inftyNorm_results["accuracy"].mean(),
                     inftyNorm_results["precision"].mean(),
                     inftyNorm_results["recall"].mean(),
                     inftyNorm_results["Error_L1"].mean(),
                     inftyNorm_results["RMS_true"].mean()]

FCLS_metrics = [FCLS_results["accuracy"].mean(),
                FCLS_results["precision"].mean(),
                FCLS_results["recall"].mean(),
                FCLS_results["Error_L1"].mean(),
                FCLS_results["RMS_true"].mean()]

LASSO_metrics = [LASSO_results["accuracy"].mean(),
                 LASSO_results["precision"].mean(),
                 LASSO_results["recall"].mean(),
                 LASSO_results["Error_L1"].mean(),
                 LASSO_results["RMS_true"].mean()]

pnrom_metrics = [pnorm_results["accuracy"].mean(),
                 pnorm_results["precision"].mean(),
                 pnorm_results["recall"].mean(),
                 pnorm_results["Error_L1"].mean(),
                 pnorm_results["RMS_true"].mean()]

metrics = ["accuracy", "precision", "recall", "error", "RMS_true"]

# create dataframe comparison
comparison_df = pd.DataFrame({"inftyNorm": inftyNorm_metrics,
                              "FCLS": FCLS_metrics,
                              "LASSO": LASSO_metrics,
                              "pnorm": pnrom_metrics},
                             index=metrics)
print(comparison_df)
comparison_df.to_csv(path + date + "labmix_comparison%s.csv" % run_num)