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
    

    ks = [-1]
    datasets = [
      'vtab/caltech101',
      'vtab/cifar10',
      'vtab/cifar100',
      'vtab/clevr_count_all',
      'vtab/clevr_closest_object_distance',
      'vtab/diabetic_retinopathy',
      'vtab/dmlab',
      'vtab/dsprites_label_orientation',
      'vtab/dsprites_label_x_position',
      'vtab/dtd',
      'vtab/eurosat',
      'vtab/kitti_closest_vehicle_distance',
      'vtab/flowers',
      'vtab/pets',
      'vtab/pcam',
      'vtab/resisc45',
      'vtab/smallnorb_label_azimuth',
      'vtab/smallnorb_label_elevation',
      'vtab/svhn',
    ]

    fig = plt.figure(constrained_layout=True, figsize=(5*3, 3.5*2))
    gs = GridSpec(2, 3, figure=fig)

    data_x = {}
    data_y = {}
    count = 0

    ax = fig.add_subplot(gs[0, 0])
    for i1 in range(len(datasets)):
        for i2 in range(1):
            
           

            pdf = df[df.dataset == datasets[i1]]
            pdf = pdf[pdf.fewshot_k == ks[i2]]


            for j, (name, groupdf) in enumerate(pdf.groupby('pretrained_clean')):
                #groupdf = groupdf.sort_values(by='lp_acc1')
                xs, ys, = [], []
                for subname, subgroupdf in groupdf.groupby('gmacs_total'):
                    xs.append(subname)
                    ys.append(subgroupdf['lp_acc1'].max())


                xs, ys = np.array(xs), np.array(ys)
                # name.replace('_e32', '') if i1 == 0 else None
                #ax.plot(xs, ys, label=None, color=f'C{j}', alpha=0.2)

                if name not in data_x:
                    data_x[name] = np.array(xs) / len(datasets)
                else:
                    data_x[name] = data_x[name] + np.array(xs) / len(datasets)
                if name not in data_y:
                    data_y[name] = np.array(ys) / len(datasets)
                else:
                    data_y[name] = data_y[name] + np.array(ys) / len(datasets)
                # lin_params, _ = linear_fit(np.log(xs), ys)
                # xs = np.log(np.array([xs.min(), xs.max()]))
                # ys = lin_params[1] * xs + lin_params[0]
                #ax.plot(np.exp(xs), ys, color=f'C{j}')

    # for j, name in enumerate([
    #     'laion2b',
    #     'laion400m_e32',
    #     'openai',
    # ]):
    for j, name in enumerate([
        'CLIP-WiT',
        'LAION',
    ]):
        print(name)
        xs, ys = data_x[name], data_y[name]
        ax.scatter(xs, ys, label=name.replace('_e32', ''), color=f'C{j}')
        lin_params, _ = linear_fit(np.log(xs), ys)
        xs = np.log(np.array([xs.min(), xs.max()]))
        ys = lin_params[1] * xs + lin_params[0]
        ax.plot(np.exp(xs), ys, color=f'C{j}')

    ax.legend()
    ax.set_xscale('log')
    # ax.set_xlabel('Activations (M)')
    ax.set_xlabel('Total compute (GMACS per sample x samples seen)')

    ax.set_ylabel('Accuracy')
    dset = datasets[i1]
    ax.set_title(f'VTAB')
    ax.grid()

    plt.savefig(
        f"probe_benchmark/vtab_scaling_plot_avg.pdf", bbox_inches="tight"
    )