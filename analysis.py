from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import matplotlib

matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})


def readResults(folder):
    results = []
    for file in Path(folder).glob('*.txt'):
        with open(file) as file:
            data = []
            for line in file.readlines():
                data.append(float(line.split('=')[1]))
            results.append(data)
    return np.array(results)

def showBarPlot(all_data):
    labels = list(range(2, 5))

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 3))

    # notch shape box plot
    bplot = ax.boxplot(all_data,
                       notch=False,  # notch shape
                       vert=True,  # vertical box alignment
                       patch_artist=True,  # fill with color
                       labels=labels)  # will be used to label x-ticks
    # ax.set_title('Notched box plot')

    # fill with colors
    colors = ['pink', 'lightblue', 'lightgreen']
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)

    # adding horizontal grid lines
    # plt.errorbar(x + 1, np.mean(y, axis=0), yerr=np.std(y, axis=0), color='k', ls='--')

    ax.yaxis.grid(True)
    ax.set_xlabel('Number of Sources')
    ax.set_ylabel('Mean Squared Error')
    plt.tight_layout()

if __name__ == '__main__':
    data = readResults('results')
    showBarPlot(data)
    # plt.show()
    plt.savefig('errorBars.pgf')