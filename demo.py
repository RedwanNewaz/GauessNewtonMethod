import  numpy as np
from scipy.optimize import least_squares

def function(x, p):
    a, b = p
    return (1 / np.sqrt(2 * np.pi * b)) * np.exp(-0.5 * (1/ b) * (x - a) ** 2)

def func(p, X, y):
    a, b = p
    return function(X, p) - y



if __name__ == '__main__':
    xvals = np.linspace(0, 5, 50)
    yvals = function(xvals, [2.5, 0.5]) + np.random.normal(0, 0.01, size=50)
    res = least_squares(func, np.array([1, 2]), args=(xvals, yvals))
    print(res)

