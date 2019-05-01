import pandas as pd
import matplotlib.pyplot as plt

#load results
path = "results/2019-03-22/"
filename = "2019-03-22_LASSO_sweep_results1"
results = pd.read_csv(path + filename + ".csv")


# display abundance error and true RMS error

fig, ax = plt.subplots()
plt.plot(results["lambda"], results["Error_L1_mean"], "r-")
plt.ylabel("Error")
plt.xlabel(r"$\lambda$")
plt.xscale("log")
plt.title("LASSO")
plt.subplots_adjust(bottom=0.15, left=0.15)
for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(15)

# plt.subplots_adjust(right=0.84)
plt.savefig(path + filename + "plot.png")
plt.show()
