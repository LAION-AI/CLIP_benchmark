from tokenize import group
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt

import statsmodels.api as sm

def linear_fit(x, y):
    x = np.array(x)
    y = np.array(y)

    covs = sm.add_constant(x, prepend=True)
    model = sm.OLS(y, covs)
    result = model.fit()
    return result.params, result.rsquared

if __name__ == '__main__':
    df = pd.read_json('probe_benchmark/scaling_experiment_data.json')
    
    ks = [10, 25, -1]
    datasets = ['imagenet1k-unverified', 'cifar100']

    fig = plt.figure(constrained_layout=True, figsize=(5*3, 3.5*2))
    gs = GridSpec(2, 3, figure=fig)

    for i1 in range(2):
        for i2 in range(3):
            ax = fig.add_subplot(gs[i1, i2])

            pdf = df[df.dataset == datasets[i1]]
            pdf = pdf[pdf.fewshot_k == ks[i2]]

            # optionally change to pretrained-short
            for j, (name, groupdf) in enumerate(pdf.groupby('pretrained_clean')):
                #groupdf = groupdf.sort_values(by='lp_acc1')
                xs, ys, = [], []
                # optionally change gmacs total to macts
                for subname, subgroupdf in groupdf.groupby('gmacs_total'):
                    xs.append(subname)
                    ys.append(subgroupdf['lp_acc1'].max())

                xs, ys = np.array(xs), np.array(ys)
                ax.scatter(xs, ys, label=name.replace('_e32', ''), color=f'C{j}')
                lin_params, _ = linear_fit(np.log(xs), ys)
                xs = np.log(np.array([xs.min(), xs.max()]))
                ys = lin_params[1] * xs + lin_params[0]
                ax.plot(np.exp(xs), ys, color=f'C{j}')

            ax.legend()
            ax.set_xscale('log')
            ax.set_xlabel('Total compute (GMACS per sample x samples seen)')
            ax.set_ylabel('Accuracy')
            dset = 'ImageNet' if i1 == 0 else 'CIFAR100'
            if ks[i2] == -1:
                ax.set_title(f'dataset = {dset}, full dataset.')
            else:
                ax.set_title(f'dataset = {dset}, {ks[i2]}-shot.')
            ax.grid()

    plt.savefig(
        f"probe_benchmark/scaling_plot.pdf", bbox_inches="tight"
    )