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

    name_to_txt = {}

    ax = fig.add_subplot(gs[0, 0])
    for i1 in range(len(datasets)):
        for i2 in range(1):
            
           

            pdf = df[df.dataset == datasets[i1]]
            pdf = pdf[pdf.fewshot_k == ks[i2]]


            for j, (name, groupdf) in enumerate(pdf.groupby('pretrained_clean')):
                #groupdf = groupdf.sort_values(by='lp_acc1')
                xs, ys, txt = [], [], []
                for subname, subgroupdf in groupdf.groupby('gmacs_total'):
                    xs.append(subname)
                    ys.append(100 * ( 1 - subgroupdf['lp_acc1'].max()))
                    txt.append(subgroupdf['model_short'].values[0])
                name_to_txt[name] = txt


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
    laion2b_legend, clip_legend, laion400m_legend = False, False, False
    for j, name in enumerate([
        'CLIP-WiT',
        'LAION',
    ]):
        print(name)
        txt = name_to_txt[name]
        xs, ys = data_x[name], data_y[name]
        #ax.scatter(xs, ys, label=name.replace('_e32', ''), color=f'C{j}')
        ###
        for i in range(len(xs)):
            print(txt[i])
            color = 'C0' if 'CLIP' in txt[i] else 'C1'
            marker = 's' if '2B' in txt[i] else 'o'
            strs = ['B/32', 'B/16','B/16+', 'L/14', 'H/14', 'g/14']
            for jj, st in enumerate(strs):
                if st in txt[i]:
                    sz = (jj+1)**1.2
            # if 'B/32' in txt[i]:
            #     sz = 1
            # elif 'B/16' in txt[i]:
            #     sz = 2
            # elif 'B/16+' in txt[i]:
            #     sz = 3
            # elif 'L/14' in txt[i]:
            #     sz = 4
            # elif 'H/14'in txt[i]:
            #     sz = 5
            # elif 'g/14'in txt[i]:
            #     sz = 6

            label = None
            if '2B' in txt[i] and not laion2b_legend:
                label = 'LAION 2B'
                laion2b_legend = True
            elif '400M' in txt[i] and not laion400m_legend:
                label = 'LAION 400M'
                laion400m_legend = True
            elif 'CLIP' in txt[i] and not clip_legend:
                label = 'CLIP WiT 400M'
                clip_legend = True
            
            if '2B' in txt[i] and 'B/32' in txt[i]:
                ax.scatter(xs[i], ys[i], marker=marker, color=color, s = 25 * sz, label=label)
                label=None
                ax.scatter(xs[i], ys[i], marker=marker, color='gray', s = 25 * sz, label='ViT-B/32')
            if '2B' in txt[i] and 'L/14' in txt[i]:
                ax.scatter(xs[i], ys[i], marker=marker, color='gray', s = 25 * sz, label='Vit-L/14')
            if '2B' in txt[i] and 'g/14' in txt[i]:
                ax.scatter(xs[i], ys[i], marker=marker, color='gray', s = 25 * sz, label='Vit-g/14')
            ax.scatter(xs[i], ys[i], marker=marker, color=color, s = 25 * sz, label=label)
        ###
        lin_params, _ = linear_fit(np.log(xs), ys)
        xs = np.log(np.array([xs.min(), xs.max()]))
        ys = lin_params[1] * xs + lin_params[0]
        ax.plot(np.exp(xs), ys, color=f'C{j}')

    ax.legend()
    dset = datasets[i1]
    ax.set_title(f'VTAB')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Total compute (GMACS per sample x samples seen)')#, fontsize=12)
    ax.set_ylabel('Error (%)')#, fontsize=12)#ax.set_ylabel('Accuracy')
    import matplotlib.ticker as mticker
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%d'))
    ax.yaxis.set_minor_formatter(mticker.ScalarFormatter())

    ax.grid()

    plt.savefig(
        f"probe_benchmark/vtab_scaling_plot_avg.pdf", bbox_inches="tight"
    )