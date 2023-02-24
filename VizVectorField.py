import sys
from VectorField import VectorField
import  numpy as np
from argparse import ArgumentParser

def viz(args):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    with open(args.file) as file:
        data = []
        for line in file.readlines():
            data.append(list(map(float, line.split(' '))))

    vf = [VectorField(item) for item in data]

    x = np.linspace(args.workspace[0], args.workspace[1], args.num_samples)
    y = np.linspace(args.workspace[0], args.workspace[1], args.num_samples)



    XX, YY = np.meshgrid(x, y)
    dim = XX.shape
    X = np.column_stack((XX.flatten(), YY.flatten()))
    y = np.array([VectorField.measurement(x, vf, isEstimation=False) for x in X])
    U, V = y.T
    UU = U.reshape(dim)
    VV = V.reshape(dim)


    width = X.ptp(axis=0)
    origin = X.min(axis=0)
    print(width, origin)
    box = np.column_stack((X.min(axis=0), X.max(axis=0))).flatten()
    print(box)
    rect = Rectangle(origin, width[0], width[1], alpha=0.6, color='skyblue')
    ax = plt.gca()
    ax.add_patch(rect)
    M = np.sqrt(UU * UU + VV * VV)  # magnitude

    plt.quiver(XX, YY, UU, VV, M, cmap=plt.cm.jet)
    plt.axis(box)
    plt.axis('off')

    plt.tight_layout()
    plt.show()
    # output = "results/" + args.file.split('/')[-1].split('.')[0] + ".eps"
    # plt.savefig(output, format='eps')

if __name__ == '__main__':
    filename = sys.argv[1]
    parser = ArgumentParser()
    parser.add_argument('--file', type=str, required=True, help='parameter files')
    parser.add_argument('--num-samples', type=int, default=15, help='number of training data')
    parser.add_argument('--noise', type=float, default=0.01, help='observation noise')
    parser.add_argument('--workspace', type=float, required=True, nargs=2)

    args = parser.parse_args()


    print(args)
    viz(args)



