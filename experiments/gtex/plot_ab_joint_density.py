import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib

font = {"size": 20}
matplotlib.rc("font", **font)
matplotlib.rcParams["text.usetex"] = True

df = pd.read_csv("./out/stan_out/cov_params_samples.csv")

plt.hist2d(df.alpha, df.lengthscale, bins=20)
plt.xlabel("$a$")
plt.ylabel("$b$")
plt.tight_layout()
plt.savefig("./out/gtex_ab_joint_plot.png", dpi=300)

plt.show()