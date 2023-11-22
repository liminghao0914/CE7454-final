import deepxde as dde
import matplotlib.pyplot as plt
import numpy as np

from mymodel import DeepONet

# Load dataset
DATADIR = "../dataset/nov_dataset/"

d = np.load(f"{DATADIR}L2_bc_train.npz", allow_pickle=True)
# d = np.load("../nov_dataset/L2_bc_train_mod.npz", allow_pickle=True)
X_train = (d["X_train0"].astype(np.float32), d["X_train1"].astype(np.float32))
y_train = d["y_train"].astype(np.float32)
print(d['X_train0'].shape)
print(d['X_train1'].shape)
print(d['y_train'].shape)

# d = np.load("../csvdata/antiderivative_unaligned_test/antiderivative_unaligned_test.npz", allow_pickle=True)
d = np.load(f"{DATADIR}L2_bc_test.npz", allow_pickle=True)
# d = np.load("../nov_dataset/L2_bc_test_mod.npz", allow_pickle=True)
X_test = (d["X_test0"].astype(np.float32), d["X_test1"].astype(np.float32))
y_test = d["y_test"].astype(np.float32)
print(d['X_test0'].shape)
print(d['X_test1'].shape)
print(d['y_test'].shape)

# The couple of the first two elements are the input, and the third element is the output. This dataset can be used with the network DeepONet for operator learning.
data = dde.data.Triple(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)

# Choose a network
m = 81
dim_x = 3

net = DeepONet(
    [m, 40, 40],
    [dim_x, 40, 40],
    "relu",
    "Glorot normal",
)

# Define a Model
model = dde.Model(data, net)

# Compile and Train
model.compile("adam", lr=0.001)
losshistory, train_state = model.train(iterations=20000)

# Plot the loss trajectory
y_pred = model.predict(X_test)
# save model
model.save("deeponet")

# print results
print("L2 relative error:", dde.metrics.l2_relative_error(y_test, y_pred))
dde.saveplot(losshistory, train_state, issave=True, isplot=True)

# save y_test and y_pred to csv
np.savetxt("y_test.csv", y_test, delimiter=",")
np.savetxt("y_pred.csv", y_pred, delimiter=",")

# save d["X_test1"]
np.savetxt("X_test1.csv", d["X_test1"], delimiter=",")