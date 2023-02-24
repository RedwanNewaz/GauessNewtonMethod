import sys
from VectorField import VectorField
import  numpy as np
from scipy.optimize import least_squares
from sklearn.metrics import mean_squared_error
from argparse import ArgumentParser

def func(P, X, Y):
    NUM_COLUMNS = 3
    NUM_ROWS = len(P) // NUM_COLUMNS
    P = P.reshape((NUM_ROWS, NUM_COLUMNS))
    vf = [VectorField(p) for p in P]
    res = [VectorField.measurement(x, vf) - y for x, y in zip(X, Y)]
    return np.array(res)

def avg_mse(gt, predicted):
    mse = mean_squared_error(gt.flatten(), predicted.flatten())
    return mse

def eval(args):
    with open(args.file) as file:
        data = []
        for line in file.readlines():
            data.append(list(map(float, line.split(' '))))

    vf = [VectorField(item) for item in data]
    X = np.random.uniform(-1, 1, (args.num_samples, 2))

    y = np.array([VectorField.measurement(x, vf) for x in X])
    # add noise
    y = y + np.random.normal(0, args.noise, size=len(X))

    gt = np.array(data)
    predict = least_squares(func, gt.flatten(), args=(X, y))
    predict = np.reshape(predict.x, gt.shape)
    return avg_mse(gt, predict)



if __name__ == '__main__':
    filename = sys.argv[1]
    parser = ArgumentParser()
    parser.add_argument('--file', type=str, required=True, help='parameter files')
    parser.add_argument('--num-samples', type=int, default=15, help='number of training data')
    parser.add_argument('--noise', type=float, default=0.01, help='observation noise')
    args = parser.parse_args()


    mse = eval(args)
    print(f"[MSE] = {mse}")




