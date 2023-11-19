import sys

import deepxde as dde
import numpy as np
from deepxde.backend import torch
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.nn import functional as F

from deepxde.nn.pytorch.fnn import FNN
from deepxde.nn.pytorch.nn import NN
from deepxde.nn import activations

from ADR_solver import solve_ADR


class Unit(nn.Module):
    def __init__(self, in_N, out_N):
        super(Unit, self).__init__()
        self.in_N = in_N
        self.out_N = out_N
        self.L = nn.Linear(in_N, out_N)

    def forward(self, x):
        x1 = self.L(x)
        x2 = torch.tanh(x1)
        return x2
    
class NN1(nn.Module):
    def __init__(self, in_N, width, depth, out_N):
        super(NN1, self).__init__()
        self.width = width
        self.in_N = in_N
        self.out_N = out_N
        self.stack = nn.ModuleList()

        self.stack.append(Unit(in_N, width))

        for i in range(depth):
            self.stack.append(Unit(width, width))

        self.stack.append(nn.Linear(width, out_N))

    def forward(self, x):
        # first layer
        for i in range(len(self.stack)):
            x = self.stack[i](x)
        return x


class NN2(nn.Module):
    def __init__(self, in_N, width, depth, out_N):
        super(NN2, self).__init__()
        self.in_N = in_N
        self.width = width
        self.depth = depth
        self.out_N = out_N

        self.stack = nn.ModuleList()

        self.stack.append(nn.Linear(in_N, width))

        for i in range(depth):
            self.stack.append(nn.Linear(width, width))

        self.stack.append(nn.Linear(width, out_N))

    def forward(self, x):
        # first layer
        for i in range(len(self.stack)):
            x = self.stack[i](x)
        return x

class DeepONetCartesianProd(NN):
    """Deep operator network for dataset in the format of Cartesian product.

    Args:
        layer_sizes_branch: A list of integers as the width of a fully connected network,
            or `(dim, f)` where `dim` is the input dimension and `f` is a network
            function. The width of the last layer in the branch and trunk net should be
            equal.
        layer_sizes_trunk (list): A list of integers as the width of a fully connected
            network.
        activation: If `activation` is a ``string``, then the same activation is used in
            both trunk and branch nets. If `activation` is a ``dict``, then the trunk
            net uses the activation `activation["trunk"]`, and the branch net uses
            `activation["branch"]`.
    """

    def __init__(
        self,
        layer_sizes_branch,
        layer_sizes_trunk,
        activation,
        kernel_initializer,
        regularization=None,
    ):
        super().__init__()
        if isinstance(activation, dict):
            activation_branch = activations.get(activation["branch"])
            self.activation_trunk = activations.get(activation["trunk"])
        else:
            activation_branch = self.activation_trunk = activations.get(activation)
        if callable(layer_sizes_branch[1]):
            # User-defined network
            self.branch = layer_sizes_branch[1]
        else:
            # Fully connected network
            self.branch = FNN(layer_sizes_branch, activation_branch, kernel_initializer)
        self.trunk = FNN(layer_sizes_trunk, self.activation_trunk, kernel_initializer)
        self.b = torch.nn.parameter.Parameter(torch.tensor(0.0))
        self.regularizer = regularization

    def forward(self, inputs):
        x_func = inputs[0]
        x_loc = inputs[1]
        # Branch net to encode the input function
        x_func = self.branch(x_func)
        # Trunk net to encode the domain of the output function
        if self._input_transform is not None:
            x_loc = self._input_transform(x_loc)
        x_loc = self.activation_trunk(self.trunk(x_loc))
        # Dot product
        if x_func.shape[-1] != x_loc.shape[-1]:
            raise AssertionError(
                "Output sizes of branch net and trunk net do not match."
            )
        x = torch.einsum("bi,ni->bn", x_func, x_loc)
        # Add bias
        x += self.b

        if self._output_transform is not None:
            x = self._output_transform(inputs, x)
        return x

def get_data_low(fname_train, fname_test):
    d = np.load(fname_train, allow_pickle=True)
    X_branch = d["X0"]
    X_trunk = d["X1"]
    y_train = d["y_low_x"]
    X_branch = np.vstack(X_branch).astype(np.float32)
    X_trunk = np.vstack(X_trunk).astype(np.float32) / 50
    y_train = np.vstack(y_train).astype(np.float32)
    X_train = (X_branch, X_trunk)

    d = np.load(fname_test, allow_pickle=True)
    X_branch = np.vstack(d["X0"]).astype(np.float32)
    X_trunk = np.vstack(d["X1"]).astype(np.float32) / 50
    y_test = np.vstack(d["y_low_x"]).astype(np.float32)
    X_test = (X_branch, X_trunk)
    return dde.data.Triple(X_train, y_train.T, X_test, y_test.T)

def get_data_high(fname_train, fname_test):
    d = np.load(fname_train, allow_pickle=True)
    X_branch = d["X0"]
    X_trunk = d["X1"]
    y_train = d["y"]
    X_branch = np.vstack(X_branch).astype(np.float32)
    X_trunk = np.vstack(X_trunk).astype(np.float32) / 50
    y_train = np.vstack(y_train).astype(np.float32)
    X_train = (X_branch, X_trunk)

    d = np.load(fname_test, allow_pickle=True)
    X_branch = np.vstack(d["X0"]).astype(np.float32)
    X_trunk = np.vstack(d["X1"]).astype(np.float32) / 50
    y_test = np.vstack(d["y"]).astype(np.float32)
    X_test = (X_branch, X_trunk)
    return dde.data.Triple(X_train, y_train.T, X_test, y_test.T)

class LowFidelityModel:
    def __init__(self, data):
        m = 50
        dim_x = 2
        width = 128

        net = dde.maps.DeepONetCartesianProd(
            [m, width, width, width],
            [dim_x, width, width, width],
            "tanh",
            "Glorot normal",
        )
        # net.apply_feature_transform(periodic_phase)
        # net.apply_output_transform(
        #     lambda _, y: y * np.std(data.train_y) + np.mean(data.train_y)
        # )
        self.model = dde.Model(data, net)
        self.model.compile("adam", lr=0)
        # import pdb; pdb.set_trace()
        # self.grad_xi = dde.grad.jacobian(net.outputs, net.inputs[0])
        self.model.restore("deeponet_dr-1000.pt", verbose=1)

    def __call__(self, X):
        return self.model.predict(X)

    def value_and_grad(self, X):
        X.requires_grad_(True)
        y = self.net(X)
        grad_y = torch.autograd.grad(y, X, create_graph=True)[0]
        return self.output_transform(y), grad_y

def get_data_multi(fname_train, fname_test, residual=False, stacktrunk=False):
    d = np.load(fname_train, allow_pickle=True)
    X_branch = d["X0"]
    X_trunk = d["X1"]
    y_train = d["y"]
    y_low_x = d["y_low_x"]
    
    X_branch = np.vstack(X_branch).astype(np.float32)
    X_trunk = np.vstack(X_trunk).astype(np.float32) / 50
    y_train = np.vstack(y_train).astype(np.float32)
    y_low_x = np.vstack(y_low_x).astype(np.float32)
    if stacktrunk:
        X_trunk = np.hstack((X_trunk, y_low_x))
    if residual:
        y_train -= y_low_x
    X_train = (X_branch, X_trunk)

    d = np.load(fname_test, allow_pickle=True)
    X_branch = np.vstack(d["X0"]).astype(np.float32)
    X_trunk = np.vstack(d["X1"]).astype(np.float32) / 50
    y_test = np.vstack(d["y"]).astype(np.float32)
    # y_low_x = np.vstack(d["y_low_x"]).astype(np.float32)  # Exact
    y_low_x = np.vstack(d["y_low_x"]).astype(np.float32)  # Model
    if stacktrunk:
        X_trunk = np.hstack((X_trunk, y_low_x))
    if residual:
        y_test -= y_low_x
    X_test = (X_branch, X_trunk)
    return dde.data.Triple(X_train, y_train.T, X_test, y_test.T)

class MultiFidelityModel:
    def __init__(self, model_low, data):
        m = 50
        dim_x = 3
        width = 128
        net = dde.maps.DeepONetCartesianProd(
            [m, width, width, width],
            [dim_x, width, width, width],
            "tanh",
            "Glorot normal",
        )
        # net.apply_feature_transform(periodic_phase)
        # net.apply_output_transform(
        #     lambda _, y: y * np.std(data.train_y) + np.mean(data.train_y)
        # )
        self.model = dde.Model(data, net)
        self.model.compile("adam", lr=0.0005)
        # self.grad_xi = dde.grad.jacobian(output_data, input_data[0])
        # self.grad_low = dde.grad.jacobian(net.register_forward_hook(get_output), self.model.data[1], j=2)
        self.model_low = model_low
        # self.model.restore("model_high/model.ckpt-398000", verbose=1)

    def __call__(self, X):
        # Check if x is inside a pore
        # in_pore = []
        # for i in range(len(X[0])):
        #     import pdb; pdb.set_trace()
        #     pores = pores_to_coordinates(X[0][i])
        #     d = np.linalg.norm(X[1][i] - pores, ord=np.inf, axis=1)
        #     in_pore.append(np.amin(d) < 0.1)
        # in_pore = np.array(in_pore)[:, None]
        # not_in_pore = 1 - in_pore

        y_low_x = self.model_low(X)
        y_pred = self.model.predict((X[0], np.hstack((X[1], y_low_x.T))))
        y_pred += y_low_x
        # return y_pred * not_in_pore
        return y_pred

    def value_and_grad(self, X):
        y_low_x, grad_xi_low = self.model_low.value_and_grad(X)
        # feed_dict = self.model.net.feed_dict(False, (X[0], np.hstack((X[1], y_low_x))))
        # y_high, grad_xi_high, grad_low = self.model.sess.run(
        #     [self.model.outputs, self.grad_xi, self.grad_low], feed_dict=feed_dict
        # )
        input_data_1 = torch.from_numpy(X[0]).float()  # device 可以是 'cpu' 或 'cuda'
        input_data_2 = torch.from_numpy(np.hstack((X[1], y_low_x))).float()

        # 调用模型并获取输出
        y_high, grad_xi_high, grad_low = self.model(input_data_1, input_data_2)
        y_high += y_low_x
        grad_xi = grad_xi_high + grad_xi_low * (1 + grad_low)
        return y_high, grad_xi
    
def test(model: dde.Model, model_low, func_space):
    # d = np.load(fname_test, allow_pickle=True)
    # X_branch = d["X0"]
    # X_trunk = d["X1"]
    # y_test = d["y"]
    
    func_feats = func_space.random(1, 1)
    xs = np.linspace(0, 1, num=100)[:, None]
    v = func_space.eval_batch(func_feats, xs)[0]
    x, t, u_true = solve_ADR(
        0,
        1,
        0,
        1,
        lambda x: 0.01 * np.ones_like(x),
        lambda x: np.zeros_like(x),
        lambda u: 0.01 * u**2,
        lambda u: 0.02 * u,
        lambda x, t: np.tile(v[:, None], (1, len(t))),
        lambda x: np.zeros_like(x),
        100,
        100,
    )
    u_true = u_true.T
    # plt.figure()
    # plt.imshow(u_true)
    # plt.colorbar()

    v_branch = func_space.eval_batch(func_feats, np.linspace(0, 1, num=50)[:, None])
    xv, tv = np.meshgrid(x, t)
    x_trunk = np.vstack((np.ravel(xv), np.ravel(tv))).T
    
    # y_low_x = d["y_low_x"]  # Exact
    err = []
    for i in range(len(v_branch)):
        # y_pred = model.predict((v_branch[i], np.hstack((X_trunk[i] / 50, y_low_x[i]))))  # Exact
        # y_pred += y_low_x[i]  # Exact
        import pdb; pdb.set_trace()
        y_low_x = model_low((v_branch[i], x_trunk[i] / 50))  # Model
        y_pred = model.predict(
            (v_branch[i], np.hstack((x_trunk[i] / 50, y_low_x)))
        )  # Model
        y_pred += y_low_x  # Model
        err.append(dde.metrics.l2_relative_error(u_true[i], y_pred))
    print(np.mean(err))

    for i in range(3):
        # y_pred = model.predict((v_branch[i], np.hstack((x_trunk[i] / 50, y_low_x[i]))))  # Exact
        # y_pred += y_low_x[i]  # Exact
        y_low_x = model_low((v_branch[i], x_trunk[i] / 50))  # Model
        y_pred = model.predict(
            (v_branch[i], np.hstack((x_trunk[i] / 50, y_low_x)))
        )  # Model
        y_pred += y_low_x  # Model
        np.savetxt(f"test{i}.dat", np.hstack((x_trunk[i], u_true[i], y_pred)))


def pores_to_coordinates(grid, N=5, L=2):
    d = L / N
    i, j = grid.reshape((N, N)).nonzero()
    x = -L / 2 + d / 2 + j * d
    y = L / 2 - d / 2 - i * d
    return np.vstack((x, y)).T


def main(args):
    # PDE
    def pde(x, y, v):
        D = 0.01
        k = 0.01
        dy_t = dde.grad.jacobian(y, x, j=1)
        dy_xx = dde.grad.hessian(y, x, j=0)
        return dy_t - D * dy_xx + k * y**2 - v


    geom = dde.geometry.Interval(0, 1)
    timedomain = dde.geometry.TimeDomain(0, 1)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)

    bc = dde.icbc.DirichletBC(geomtime, lambda _: 0, lambda _, on_boundary: on_boundary)
    ic = dde.icbc.IC(geomtime, lambda _: 0, lambda _, on_initial: on_initial)

    pde = dde.data.TimePDE(
        geomtime,
        pde,
        [bc, ic],
        num_domain=200,
        num_boundary=40,
        num_initial=20,
        num_test=500,
    )

    # Function space
    func_space = dde.data.GRF(length_scale=0.2)

    # Data
    eval_pts = np.linspace(0, 1, num=50)[:, None]
    data_pde = dde.data.PDEOperatorCartesianProd(
        pde, func_space, eval_pts, 1000, function_variables=[0], num_test=100, batch_size=100
    )
    data_low = get_data_low("low_dr_data_train.npz", "low_dr_data_test.npz")
    data_mf = get_data_multi("mf_dr_data_train.npz", fname_test="mf_dr_data_test.npz", residual=True, stacktrunk=True)
    
    data_high = get_data_high("mf_dr_data_train.npz", "mf_dr_data_test.npz")
    m = 50
    dim_x = 3
    width = 128
    net = dde.maps.DeepONetCartesianProd(
        [m, width, width, width],
        [dim_x, width, width, width],
        "tanh",
        "Glorot normal",
    )
    # net.apply_feature_transform(periodic_phase)
    # net.apply_output_transform(
    #     lambda _, y: y * np.std(data.train_y) + np.mean(data.train_y)
    # )
    data_mf_with_pde = data_pde
    data_mf_with_pde.train_x = data_mf.train_x
    data_mf_with_pde.train_y = data_mf.train_y
    data_mf_with_pde.test_x = data_mf.test_x
    data_mf_with_pde.test_y = data_mf.test_y
    
    model_low = LowFidelityModel(data_low)
    model_mf = MultiFidelityModel(model_low, data_mf)
    model = dde.Model(data_mf_with_pde, net)
    model.compile("adam", lr=0.0005)
    
    # checkpointer = dde.callbacks.ModelCheckpoint(
    #     "model/model.ckpt", save_better_only=True, period=1000
    # )
    # net_h = DeepONetCartesianProd(
    #     [m, width, width, width],
    #     [dim_x, width, width, width],
    #     "tanh",
    #     "Glorot normal",
    # )
    # net_d = DeepONetCartesianProd(
    #     [m, width, width, width],
    #     [dim_x, width, width, width],
    #     "tanh",
    #     "Glorot normal",
    # )
    # train_mf(net_d, net_h, data_low, data_high)
    epochs = 1000
    if args.mode == "train":
        losshistory, train_state = model.train(
            epochs=epochs,
            # callbacks=[checkpointer],
            display_every=100
        )
        model.save("mf_deeponet_dr")
    elif args.mode == "test":
        model_mf.model.restore(f"mf_deeponet_dr-{epochs}.pt", verbose=1)
        model_low.model.restore(f"low_deeponet_dr-{800}.pt", verbose=1)
    # dde.postprocessing.save_loss_history(losshistory, "loss.dat")
    # test(model, model_low, func_space)

    # Test
    # Function space
    func_space = dde.data.GRF(length_scale=0.2)
    
    func_feats = func_space.random(1, 1)
    xs = np.linspace(0, 1, num=100)[:, None]
    v = func_space.eval_batch(func_feats, xs)[0]
    x, t, u_true = solve_ADR(
        0,
        1,
        0,
        1,
        lambda x: 0.01 * np.ones_like(x),
        lambda x: np.zeros_like(x),
        lambda u: 0.01 * u**2,
        lambda u: 0.02 * u,
        lambda x, t: np.tile(v[:, None], (1, len(t))),
        lambda x: np.zeros_like(x),
        100,
        100,
    )
    u_true = u_true.T
    # plt.figure()
    # plt.imshow(u_true)
    # plt.colorbar()

    v_branch = func_space.eval_batch(func_feats, np.linspace(0, 1, num=int(50))[:, None])
    xv, tv = np.meshgrid(x, t)
    x_trunk = np.vstack((np.ravel(xv), np.ravel(tv))).T
    u_pred_mf = model_mf((v_branch, x_trunk))
    u_pred_low = model_low((v_branch, x_trunk))
    
    # import pdb; pdb.set_trace()
    u_pred_mf = u_pred_mf.reshape((100, 100))
    u_pred_low = u_pred_low.reshape((100, 100))
    print(dde.metrics.l2_relative_error(u_true, u_pred_mf))

    u_error = np.abs(u_true - u_pred_mf)

    fig = plt.figure(figsize=(18,5)) 
    plt.figure()
    # plt.imshow(u_pred)
    # plt.colorbar()
    # plt.show()
    # plt.savefig("u_pred.png")
    # plt.subplot(1,2,1)
    # plt.imshow(u_true, cmap='jet')
    # plt.title('Ground True')
    # # plt.colorbar()
    # plt.tight_layout()

    # plt.subplot(1,2,1)
    # plt.imshow(u_pred_low, cmap='jet')
    # plt.title('Predict_low')
    # # plt.colorbar()
    # plt.tight_layout()

    # plt.subplot(1,2,2)
    # plt.imshow(u_error, cmap='jet')
    # plt.title('Absolute error')
    # # plt.colorbar()
    # plt.tight_layout()
    # plt.savefig("3diffusion_reaction_low.png")
    
    # plt.figure()
    # plt.imshow(u_pred)
    # plt.colorbar()
    # plt.show()
    # plt.savefig("u_pred.png")
    # plt.subplot(1,2,1)
    # plt.imshow(u_true, cmap='jet')
    # plt.title('Ground True')
    # # plt.colorbar()
    # plt.tight_layout()

    plt.subplot(1,2,1)
    plt.imshow(u_pred_mf, cmap='jet')
    plt.title('Predict')
    # plt.colorbar()
    plt.tight_layout()

    plt.subplot(1,2,2)
    plt.imshow(u_error, cmap='jet')
    plt.title('Absolute error')
    # plt.colorbar()
    plt.tight_layout()
    plt.savefig("3diffusion_reaction_mf.pdf")
    
    

    
if __name__ == "__main__":
    import argparse
    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="train")
    args = parser.parse_args()
    
    main(args)
