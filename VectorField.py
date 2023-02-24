import numpy as np
class VectorField:
    def __init__(self, param):
        self.l = param[0]
        self.source =np.array(param[1:])

    def __call__(self, x, y):
        A = np.zeros((2, 2))
        A[0, 0] = (self.l - 1) * x ** 2 - y ** 2
        A[0, 1] = self.l * x * y
        A[1, 0] = self.l * x * y
        A[1, 1] = (self.l - 1) * y ** 2 - x ** 2
        F = A @ self.source

        return F

    def __repr__(self):
        return f"type = {self.l} source = {self.source}"

    @staticmethod
    def measurement(p: np.ndarray, vfList:list, isEstimation=True):
        y = np.array([v(p[0], p[1]) for v in vfList])
        # print(y.shape)
        return y.sum(axis=0) if not isEstimation else y.sum()


