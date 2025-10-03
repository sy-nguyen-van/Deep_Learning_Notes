import numpy as np
x = np.array([200,17])
w1_1 = np.array([1,2])
b1_1 = np.array([-1])
z1_1 = np.dot(w1_1,x) + b1_1
def g(x):
    """Numerically stable sigmoid function.

    Uses a branch to avoid overflow for large positive/negative inputs and
    works with scalars or numpy arrays.
    """
    x = np.asarray(x)
    # For x >= 0: sigmoid = 1 / (1 + exp(-x))
    # For x <  0: sigmoid = exp(x) / (1 + exp(x)) to avoid overflow in exp(-x)
    positive = x >= 0
    out = np.empty_like(x, dtype=float)
    out[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    exp_x = np.exp(x[~positive])
    out[~positive] = exp_x / (1.0 + exp_x)
    return out

def dense(a_in,W,b):
    units = W.shape[1]
    a_out = np.zeros(units)
    for j in range(units):
        w= W[:,j]
        z=np.dot(w,a_in) + b[j]
        a_out[j] = g(z) # sigmoid function
    return a_out
# --------------
def sequential(x):
    a1 = dense(x,W1,b1)
    a2 = dense(a1,W2,b2)
    a3 = dense(a2,W3,b3)
    a4 = dense(a3,W4,b4)
    f_x =a4
    return f_x

# Capital W --> Matrix


if __name__ == "__main__":
    # Quick checks / demo for the sigmoid implementation
    print("z1_1:", z1_1)
    print("sigmoid(z1_1):", g(z1_1))
    print("sigmoid(x):", g(x))
    # show behavior on extreme values to illustrate numerical stability
    extreme = np.array([1000.0, -1000.0, 0.0])
    print("sigmoid(extreme):", g(extreme))

ANI: Artificial narrow intelligence: Smart speaker, self-driving car, web search
AGI: Artificial general intelligence: Do anything
We have almost no idea how the brain works
Can we mimic the human brain 












