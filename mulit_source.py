import  numpy as np
from scipy.optimize import least_squares
from collections import namedtuple

vf_param = namedtuple("vf_param", "l a b")
vf_param_to_np = lambda x:np.array([x.l, x.a, x.b], dtype=np.float64)
def vector_field(l, x, y, a, b):
    p = np.array([a, b])
    A = np.zeros((2, 2))
    A[0, 0] = (l - 1) * x ** 2 - y ** 2
    A[0, 1] = l * x * y
    A[1, 0] = l * x * y
    A[1, 1] = (l - 1) * y ** 2 - x ** 2
    return np.sum(A @ p)

def function(X, p):
    result = []
    P = p.reshape((3, 3))
    for x in X:
        res = 0
        for p in P:
            l, a, b = p
            res += vector_field(l, x[0], x[1], a, b)
        result.append(res)
    return np.array(result)

def func(p, X, y):
    return function(X, p) - y


if __name__ == '__main__':
    parameters = [vf_param(l=1, a=0, b=1),
                  vf_param(l=2, a=2, b=4),
                  vf_param(l=1, a=-1, b=3)
                  ]
    np_params = np.array(list(map(vf_param_to_np, parameters)))
    print('Ground Truth Parameters \n', np_params)

    # generate samples
    X = np.random.uniform(-1, 1, 100)
    Y = np.random.uniform(-1, 1, 100)

    xvals = np.column_stack([X, Y])
    yvals = function(xvals, np_params.flatten()) + np.random.normal(0, 0.01, size=len(xvals))

    res = least_squares(func, np_params.flatten(), args=(xvals, yvals))
    print('Estimated Parameters\n',  np.reshape(res.x, (3, 3)))