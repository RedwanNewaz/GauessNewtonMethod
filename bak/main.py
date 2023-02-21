import  numpy as np
from scipy.optimize import least_squares

def vector_field(l, x, y, a, b):
    p = np.array([a, b])
    A = np.zeros((2, 2))
    A[0, 0] = (l - 1) * x ** 2 - y ** 2
    A[0, 1] = l * x * y
    A[1, 0] = l * x * y
    A[1, 1] = (l - 1) * y ** 2 - x ** 2
    return np.sum(A @ p)

def function(X, p):
    a, b = p
    res = map(lambda x: vector_field(l, x[0], x[1], a, b), X)
    return np.array(list(res))

def func(p, X, y):
    return function(X, p) - y

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # attractive vector field p = [1, 0] and l = 2
    p = np.array([1, 0])
    l = 2

    # generate samples
    X = np.random.uniform(-1, 1, 100)
    Y = np.random.uniform(-1, 1, 100)

    xvals = np.column_stack([X, Y])
    yvals = function(xvals, [2.5, 0.5]) + np.random.normal(0, 0.01, size=len(xvals))

    res = least_squares(func, np.array([1, 2]), args=(xvals, yvals))
    print(res)


