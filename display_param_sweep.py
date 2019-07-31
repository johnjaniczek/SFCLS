import pandas as pd
import matplotlib.pyplot as plt

#load results
date = "2019-07-11"
run_num = "0"
path = "results/" + date + "/"

# LASSO
filename = date + "_LASSO_sweep_results" + run_num + ".csv"
LASSO_results = pd.read_csv(path + filename)

# infty
filename = date + "_inftyNorm_sweep_results" + run_num + ".csv"
infty_results = pd.read_csv(path + filename)

# pnorm
filename = date + "_p_norm_sweep_results" + run_num + ".csv"
pnorm_results = pd.read_csv(path + filename)

# extract unique ps
p_range = pnorm_results["p"].unique()

# display abundance L1 error vs lambda
fig, ax = plt.subplots()
# for each p make a different line
for p in p_range:
    plt.plot(pnorm_results["lambda"][pnorm_results["p"] == p],
             pnorm_results["Error_L1_mean"][pnorm_results["p"] == p],
             label=r"$L_p, p=$%.4f" % p)

# plot LASSO
plt.plot(LASSO_results["lambda"],
             LASSO_results["Error_L1_mean"],
             label="LASSO")

# plot inftyNorm
plt.plot(infty_results["lambda"],
             infty_results["Error_L1_mean"],
             label=r"$L_{\infty}^{-1}$")

plt.legend()
plt.ylabel(r"$error_x$", fontweight='bold')
plt.xlabel(r"$\lambda$", fontweight='bold')
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
plt.xscale("log")
plt.ylim((0.0, 0.04))
plt.title(r"Hyperparameter Sweep", fontweight='bold')
plt.subplots_adjust(bottom=0.15, left=0.18)
for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(15)

plt.savefig('results/plots/noise_sweep_plot.png')
plt.show()

