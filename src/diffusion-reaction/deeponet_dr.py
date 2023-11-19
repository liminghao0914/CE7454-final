"""Backend supported: tensorflow.compat.v1, tensorflow, pytorch, paddle"""
import deepxde as dde
import matplotlib.pyplot as plt
import numpy as np

from ADR_solver import solve_ADR

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
    data = dde.data.PDEOperatorCartesianProd(
        pde, func_space, eval_pts, 1000, function_variables=[0], num_test=100, batch_size=100
    )

    # Net
    net = dde.nn.DeepONetCartesianProd(
        [50, 128, 128, 128],
        [2, 128, 128, 128],
        "tanh",
        "Glorot normal",
    )

    # import pdb; pdb.set_trace()
    model = dde.Model(data, net)
    model.compile("adam", lr=0.0005)
    # import pdb; pdb.set_trace()
    epochs = 1000
    if args.mode == "train":
        losshistory, train_state = model.train(iterations=epochs, display_every=10)
        dde.utils.plot_loss_history(losshistory)
        # save
        model.save("deeponet_dr")
    elif args.mode == "test":
        model.restore(f"deeponet_dr-{epochs}.pt")

    # Test
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
    u_pred = model.predict((v_branch, x_trunk))
    # save low fidelity data
    # save_lfdata(v_branch, x_trunk, u_true, u_pred)
    sample_test(model, func_space)
    # import pdb; pdb.set_trace()
    u_pred = u_pred.reshape((100, 100))
    print(dde.metrics.l2_relative_error(u_true, u_pred))

    u_error = np.abs(u_true - u_pred)

    fig = plt.figure(figsize=(18,5)) 
    plt.figure()
    # plt.imshow(u_pred)
    # plt.colorbar()
    # plt.show()
    # plt.savefig("u_pred.png")
    # plt.subplot(1,3,1)
    # plt.imshow(u_true, cmap='jet')
    # plt.title('Ground True')
    # # plt.colorbar()
    # plt.tight_layout()

    plt.subplot(1,2,1)
    plt.imshow(u_pred, cmap='jet')
    plt.title('Predict')
    # plt.colorbar()
    plt.tight_layout()

    plt.subplot(1,2,2)
    plt.imshow(u_error, cmap='jet')
    plt.title('Absolute error')
    # plt.colorbar()
    plt.tight_layout()
    plt.savefig("3diffusion_reaction_don.pdf")

def sample_test(model, func_space):
    sample_num = 1100
    for i in range(sample_num):
        func_feats = func_space.random(1, i)
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
        u_pred = model.predict((v_branch, x_trunk))
        # concat all v_branch
        if i == 0:
            v_branch_to_save = v_branch
        else:
            v_branch_to_save = np.vstack((v_branch_to_save, v_branch),)
    # save low fidelity data
    save_lfdata(v_branch_to_save, x_trunk, u_true, u_pred)

def save_lfdata(v_branch, x_trunk, y, y_pred):
    np.savez(f"mf_dr_data_train.npz", X0=v_branch[:1000,:], X1=x_trunk[:320,:], y=y.reshape(-1,1)[:320,:], y_low_x=y_pred.T[:320,:])
    np.savez(f"mf_dr_data_test.npz", X0=v_branch[-100:,:], X1=x_trunk[-566:,:], y=y.reshape(-1,1)[-566:,:], y_low_x=y_pred.T[-566:,:])
    print(f"shapes: {v_branch[:1000,:].shape}, {x_trunk[:320,:].shape}, {y.reshape(-1,1)[:320,:].shape}, {y_pred.T[:320,:].shape}")
    
    
if __name__ == "__main__":
    import argparse
    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="train")
    args = parser.parse_args()
    
    main(args)