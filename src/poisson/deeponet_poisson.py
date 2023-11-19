import sys

import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt
from spaces import GRF

def get_data(
    fname_train, fname_test, residual=False, stackbranch=False, stacktrunk=False
):
    N = 50000
    i = 0
    idx = np.arange(i * N, (i + 1) * N)
    idx = np.random.choice(100000, size=N, replace=False)
    idx = np.arange(N)

    d = np.load(fname_train)
    X_branch = d["X0"][idx].astype(np.float32)
    X_trunk = d["X1"].astype(np.float32)
    if stackbranch:
        X_branch = np.hstack((d["X0"][idx].astype(np.float32), d["y_low"][idx].astype(np.float32)))
    if stacktrunk:
        X_trunk = np.hstack((d["X1"][idx].astype(np.float32), d["y_low_x"][idx].astype(np.float32)))
    X_train = (X_branch, X_trunk)
    y_train = d["y"][idx].astype(np.float32)
    if residual:
        y_train -= d["y_low_x"][idx].astype(np.float32)
    # d = np.load(fname_train)
    # X_branch = d["X0"].astype(np.float32)
    # X_trunk = d["X1"].astype(np.float32)
    # if stackbranch:
    #     X_branch = np.hstack((d["X0"].astype(np.float32), d["y_low"].astype(np.float32)))
    # if stacktrunk:
    #     X_trunk = np.hstack((d["X1"].astype(np.float32), d["y_low_x"].astype(np.float32)))
    # X_train = (X_branch, X_trunk)
    # y_train = d["y"].astype(np.float32)
    # if residual:
    #     y_train -= d["y_low_x"].astype(np.float32)

    d = np.load(fname_test)
    X_branch = d["X0"].astype(np.float32)
    X_trunk = d["X1"].astype(np.float32)
    if stackbranch:
        X_branch = np.hstack((d["X0"].astype(np.float32), d["y_low"].astype(np.float32)))
    if stacktrunk:
        X_trunk = np.hstack((d["X1"].astype(np.float32), d["y_low_x"].astype(np.float32)))
    X_test = (X_branch, X_trunk)
    y_test = d["y"].astype(np.float32)
    if residual:
        y_test -= d["y_low_x"].astype(np.float32)
    return X_train, y_train, X_test, y_test


def run(data, net, lr, epochs):
    model = dde.Model(data, net)
    model.compile("adam", lr=lr)
    losshistory, train_state = model.train(epochs=epochs)
    dde.saveplot(losshistory, train_state, issave=False, isplot=True)
    model.save("mf_model")


def main(args):
    fname_train = "train.npz"
    fname_test = "test.npz"
    X_train, y_train, X_test, y_test = get_data(
        fname_train, fname_test, residual=False, stackbranch=False, stacktrunk=False
    )
    data = dde.data.Triple(
        X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    m = 100
    dim_x = 1
    width = 32
    net = dde.maps.DeepONetCartesianProd(
        [m, width, width],
        [dim_x, width, width],
        "tanh",
        "Glorot normal",
    )

    lr = 0.0001
    epochs = 10000
    model = dde.Model(data, net)
    dde.optimizers.set_LBFGS_options(maxiter=epochs)
    model.compile("L-BFGS")
    
    # import pdb; pdb.set_trace()
    if args.mode == "train":
        losshistory, train_state = model.train(epochs=epochs, display_every=100)
        dde.saveplot(losshistory, train_state, issave=True, isplot=True)
        model.save("poisson")
    elif args.mode == "test":
        model.restore(f"poisson-{epochs}.pt")
        
        # # Domain is interval [0, 1]
        # geom = dde.geometry.Interval(0, 1)
        # # Function space for f(x) are polynomials
        # degree = 3
        # space = dde.data.PowerSeries(N=degree + 1)
        # # Choose evaluation points
        # num_eval_points = 100
        # evaluation_points = geom.uniform_points(num_eval_points, boundary=True)
        # num = 3
        # features = space.random(num)
        # fx = space.eval_batch(features, evaluation_points)
        # x = geom.uniform_points(100, boundary=True)
        
        space = GRF(1, length_scale=0.05, N=1000, interp="cubic")
        m = 100
        num = 100000

        features = space.random(num)
        x = np.linspace(0, 1, num=m)[:, None]
        # fx = space.eval_u(features, x)
        # get from dataset
        d = np.load(fname_test)
        fx = d["X0"].astype(np.float32)
        # x = []
        # for i in range(m):
        #     idx = np.random.randint(0, m, size=1)
        #     x.append(sensors[idx])
        # x = np.array(x)
        
        y = model.predict((fx, x))
        # Setup figure
        fig = plt.figure(figsize=(7, 8))
        plt.subplot(2, 1, 1)
        plt.title("Poisson equation: Source term f(x) and solution u(x)")
        plt.ylabel("f(x)")
        z = np.zeros_like(x)
        plt.plot(x, z, "k-", alpha=0.1)
        # Plot source term f(x)
        # for i in range(3,5):
        #     plt.plot(x, fx[i], "--")
        plt.plot(x, fx[4], "--")
        
        # Plot solution u(x)
        plt.subplot(2, 1, 2)
        plt.ylabel("u(x)")
        plt.plot(x, z, "k-", alpha=0.1)
        # TODO: mean y[i]
        # y_mean = []
        # for i in range(num):
        #     y_mean.append(np.mean(y[i]))
        # y_mean = np.array(y_mean)
        plt.plot(x, y[4], "-")
        plt.xlabel("x")

        # plt.show()
        plt.savefig("poisson.png")
        np.savez_compressed("poisson.npz", x=x, y=y, fx=fx)
        import pdb; pdb.set_trace()
        # print(dde.metrics.l2_relative_error(y_test, y_pred))


if __name__ == "__main__":
    import argparse
    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="train")
    args = parser.parse_args()
    
    main(args)
