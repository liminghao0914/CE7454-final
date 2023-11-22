import matplotlib.pyplot as plt
import numpy as np

mf_data = np.load("saved/results/poisson/mf_poisson.npz")
high_data = np.load("saved/results/poisson/high_poisson.npz")
low_data = np.load("saved/results/poisson/low_poisson.npz")
norm_data = np.load("saved/results/poisson/poisson.npz")

xs = [mf_data["x"], high_data["x"], low_data["x"], norm_data["x"]]
ys = [mf_data["y"], high_data["y"], low_data["y"], norm_data["y"]]
fxs = [mf_data["fx"], high_data["fx"], low_data["fx"], norm_data["fx"]]

fig = plt.figure(figsize=(7, 5))
plt.subplot(2, 1, 1)
# plt.title("Poisson equation: Source term f(x) and solution u(x)", fontsize=18)
plt.ylabel("f(x)", fontsize=18)
z = np.zeros_like(mf_data["x"])
plt.plot(norm_data["x"], z, "k-", alpha=0.1)
plt.plot(mf_data["x"][:,0], mf_data["fx"][4][:100], "r-")
plt.plot(high_data["x"], high_data["fx"][4], "b--")
plt.plot(low_data["x"], low_data["fx"][4], "o--")
plt.plot(norm_data["x"], norm_data["fx"][4], "k-", label="Normal")
plt.legend(loc="lower right", fontsize=12)
plt.grid()

# Plot solution u(x)
plt.subplot(2, 1, 2)
plt.ylabel("u(x)", fontsize=12)
plt.plot(norm_data["x"], z, "k-", alpha=0.1)
plt.plot(mf_data["x"][:,0], mf_data["y"][4][:100], "r-", label="MF")
plt.plot(high_data["x"], high_data["y"][4], "b--", label="HF")
plt.plot(low_data["x"], low_data["y"][4], "o--", label="LF")
plt.plot(norm_data["x"], norm_data["y"][4], "k-")
plt.xlabel("x", fontsize=12)
plt.legend(loc="upper right", fontsize=12)
plt.grid()
# plt.legend()

# plt.show()
plt.savefig("poisson.pdf")