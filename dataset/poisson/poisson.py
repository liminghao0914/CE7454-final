import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate

from spaces import GRF
from utils import timing
import random

def solver(f, N):
    """u_xx = 20f, x \in [0, 1]
    u(0) = u(1) = 0
    """
    h = 1 / (N - 1)
    K = -2 * np.eye(N - 2) + np.eye(N - 2, k=1) + np.eye(N - 2, k=-1)
    b = h ** 2 * 20 * f[1:-1]
    u = np.linalg.solve(K, b)
    u = np.concatenate(([0], u, [0]))
    return u


def example():
    space = GRF(1, length_scale=0.05, N=1000, interp="cubic")
    m = 100

    features = space.random(1)
    sensors = np.linspace(0, 1, num=m)
    sensor_values = space.eval_u(features, sensors[:, None])
    y = solver(sensor_values[0], m)
    np.savetxt("poisson_high.dat", np.vstack((sensors, np.ravel(sensor_values), y)).T)

    m_low = 10
    x_low = np.linspace(0, 1, num=m_low)
    f_low = space.eval_u(features, x_low[:, None])
    y_low = solver(f_low[0], m_low)
    np.savetxt("poisson_low.dat", np.vstack((x_low, y_low)).T)


@timing
def gen_data():
    print("Generating operator data...", flush=True)
    space = GRF(1, length_scale=0.05, N=1000, interp="cubic")
    m = 100
    num = 1000

    features = space.random(num)
    sensors = np.linspace(0, 1, num=m)
    sensor_values = space.eval_u(features, sensors[:, None])
    x = []
    y = []
    for i in range(num):
        tmp = solver(sensor_values[i], m)
        tmp = interpolate.interp1d(sensors, tmp, copy=False, assume_sorted=True)
        idx = np.random.randint(0, m, size=1)
        x.append(sensors[idx])
        y.append(tmp(sensors))
    x = np.array(x)
    y = np.array(y)

    m_low = 10
    x_low = np.linspace(0, 1, num=m_low)
    f_low = space.eval_u(features, x_low[:, None])
    y_low = []
    y_low_x = []
    for i in range(num):
        tmp = solver(f_low[i], m_low)
        tmp = interpolate.interp1d(x_low, tmp, copy=False, assume_sorted=True)
        y_low.append(tmp(x_low))
        y_low_x.append(tmp(x[i]))
    y_low = np.array(y_low)
    y_low_x = np.array(y_low_x)
    
    x1 = sensors[:, np.newaxis]
    x1_low = x_low[:, np.newaxis]
    y_low_x = tmp(sensors)[:, np.newaxis]
    np.savez_compressed(
        "test.npz", X0=sensor_values, X1=x1, y=y, y_low=y_low, y_low_x=y_low_x, x1_low=x1_low, f_low=f_low
    )
    # import pdb; pdb.set_trace()
    plt.rcParams['font.size'] = 18
    # for i in range(5):
    plt.figure()
    plt.plot(sensors, sensor_values[4], "r-", label="High-fidelity")
    # plt.plot(x[i], y[i], "or")
    plt.plot(x_low, f_low[4], "o--", label="Low-fidelity")
    # plt.plot(x[i], y_low_x[i], "xb")
    # plt.show()
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    plt.savefig("poisson_f.pdf")
    
    # import pdb; pdb.set_trace()
    plt.figure()
    plt.plot(sensors, solver(sensor_values[4], m), "r-", label="High-fidelity")
    plt.plot(x_low, solver(f_low[4], m_low), "o--", label="Low-fidelity")
    plt.xlabel("x")
    plt.ylabel("u(x)")
    plt.legend()
    plt.savefig("poisson_u.pdf")

def main():
    # example()
    gen_data()


if __name__ == "__main__":
    main()
