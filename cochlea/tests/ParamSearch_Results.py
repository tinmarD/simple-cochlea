import numpy as np
import seaborn as sns
import pandas as pd
from multiprocessing import Pool
from bokeh.io import output_file, show
from bokeh.plotting import figure
from bokeh.layouts import column
from bokeh.models import LinearAxis, Range1d, HoverTool, Legend
from cochlea import *


def plot_results(df, sort_key):
    # Get each model of cochlea and its index in the DataFrame
    grp = df.groupby(['fmin', 'fmax', 'n_channels', 'freq_scale', 'fbank_type', 'forder', 'comp_factor', 'comp_gain',
                      'lif_tau_coeff', 'lif_v_thresh_start', 'lif_v_thresh_stop']).indices
    grp_index = list(grp.values())
    # Compute the mean scores across the differents sounds for each of these cochlea models
    n_models = len(grp_index)
    n_sounds = [len(index_i) for index_i in grp_index]
    if not np.unique(n_sounds).size == 1:
        print('Warning: some cochlea models appears more than others')
    df_mean = df.groupby(['fmin', 'fmax', 'n_channels', 'freq_scale', 'fbank_type', 'forder', 'comp_factor', 'comp_gain',
                           'lif_tau_coeff', 'lif_v_thresh_start', 'lif_v_thresh_stop']).mean().reset_index()
    df_mean = df_mean.sort_values(sort_key)
    # Use bokeh to create interactive figure
    # 1st subplot : vr_score and spkchanvar_score
    p1 = figure(title='Cochlea Best Parameter Search', y_range=(-0.1, 1.1), plot_width=1200, plot_height=400)
    p1.line(np.arange(0, n_models), 'channel_with_spikes_ratio', source=df_mean, color='green', alpha=0.6)
    c13 = p1.circle(np.arange(0, n_models), 'channel_with_spikes_ratio', source=df_mean, color='green', alpha=0.6)
    r1 = p1.line(np.arange(0, n_models), 'vr_score', source=df_mean)
    c11 = p1.circle(np.arange(0, n_models), 'vr_score', source=df_mean)
    p1.extra_y_ranges = {"foo": Range1d(start=0, end=5)}
    p1.line(np.arange(0, n_models), 'spkchanvar_score', source=df_mean, color='red', y_range_name="foo")
    c12 = p1.circle(np.arange(0, n_models), 'spkchanvar_score', source=df_mean, color='red', y_range_name="foo")
    p1.add_layout(LinearAxis(y_range_name="foo"), 'right')
    p1.add_layout(Legend(items=[('Van Rossum score', [c11]), ('CV(spikes_per_channel)', [c12]),
                                ('Channel with Spikes', [c13])]), 'left')
    # 2nd subplot : Global Score and mean number of spikes
    p2 = figure(y_range=(-0.1, 10),  plot_width=1200, plot_height=400)
    p2.x_range = p1.x_range
    r2 = p2.line(np.arange(0, n_models), 'global_score', source=df_mean)
    c21 = p2.circle(np.arange(0, n_models), 'global_score', source=df_mean)
    p2.extra_y_ranges = {"foo": Range1d(start=0, end=40000)}
    p2.line(np.arange(0, n_models), 'n_spikes_mean', source=df_mean, color='orange', y_range_name="foo")
    c22 = p2.circle(np.arange(0, n_models), 'n_spikes_mean', source=df_mean, color='orange', y_range_name="foo")
    p2.add_layout(LinearAxis(y_range_name="foo"), 'right')
    p2.add_layout(Legend(items=[('Global Score', [c21]), ('Number of spikes', [c22])]), 'left')
    # Add Hover Tools
    hover_1 = HoverTool(tooltips=[('Number of channels', '@n_channels'), ('Filters ', '@forder order @fbank_type'),
                                  ('Freq. Range', '[@fmin - @fmax] Hz'), ('Comp. factor', '@comp_factor'),
                                  ('Comp. gain', '@comp_gain'), ('LIF Tau coeff', '@lif_tau_coeff'),
                                  ('LIF threshold', '[@lif_v_thresh_start - @lif_v_thresh_stop]')],
                        renderers=[r1], mode='vline')
    p1.add_tools(hover_1)
    hover_2 = HoverTool(tooltips=[('Global Score', '@global_score'), ('Van-Rossum Score', '@vr_score'),
                                  ('CV(chan_spikes)', '@spkchanvar_score'),
                                  ('Channel with spikes', '@channel_with_spikes_ratio (@n_spikes_mean spikes)')],
                        renderers=[r2], mode='vline')
    p2.add_tools(hover_2)
    layout = column(p1, p2)
    output_file('cochlea_parameter_search_output.html')
    show(layout)


def plot_param_score(df, param_key=[], inner='box', xticks_rotation=0):
    param_key = np.atleast_1d(param_key)
    if param_key.size == 0:
        param_key = np.array(['n_channels', 'freq_scale', 'fbank_type', 'forder', 'fmin', 'fmax', 'comp_factor',
                              'comp_gain', 'lif_tau_coeff', 'lif_v_thresh_start', 'lif_v_thresh_stop'])
    if param_key.size > 1:
        for param_key_i in param_key:
            plot_param_score(df, param_key=param_key_i, inner=inner)
    elif param_key.size == 1:
        param_key = str(param_key[0])
        f = plt.figure()
        ax0 = plt.subplot2grid((2, 6), (0, 0), colspan=2)
        sns.violinplot(x=param_key, y='global_score', data=df, inner=inner, ax=ax0)
        plt.xticks(rotation=xticks_rotation)
        ax1 = plt.subplot2grid((2, 6), (0, 2), colspan=2)
        sns.violinplot(x=param_key, y='vr_score', data=df, inner=inner, ax=ax1)
        plt.xticks(rotation=xticks_rotation)
        ax2 = plt.subplot2grid((2, 6), (0, 4), colspan=2)
        sns.violinplot(x=param_key, y='spkchanvar_score', data=df, inner=inner, ax=ax2)
        plt.xticks(rotation=xticks_rotation)
        ax3 = plt.subplot2grid((2, 6), (1, 1), colspan=2)
        sns.violinplot(x=param_key, y='channel_with_spikes_ratio', data=df, inner=inner, ax=ax3)
        plt.xticks(rotation=xticks_rotation)
        ax4 = plt.subplot2grid((2, 6), (1, 3), colspan=2)
        sns.violinplot(x=param_key, y='n_spikes_mean', data=df, inner=inner, ax=ax4)
        plt.xticks(rotation=xticks_rotation)
        ax1.set(title='Cochlea parameter : {}'.format(param_key))

if __name__ == '__main__':
    # Input sig
    results_filepath = 'C:\\Users\\deudon\\Desktop\\M4\\_Results\\Python_cochlea\\ParamSearch_results.csv'
    df = pd.read_csv(results_filepath, sep=';')
    df.keys()
    df = df.drop('Unnamed: 0', axis=1)
    # global_score, vr_score, spkchanvar_score, channel_with_spikes_ratio, n_spikes_mean = \
    #     map(lambda x: np.array(df.get(x)), ['global_score', 'vr_score', 'spkchanvar_score', 'channel_with_spikes_ratio',
    #                                         'n_spikes_mean'])
    plot_param_score(df)

    # df_sel = df.get(['global_score', 'vr_score', 'spkchanvar_score', 'channel_with_spikes_ratio', 'n_spikes_mean'])
    # nan_rows = np.unique(np.where(np.isnan(df_sel.values))[0])
    # if nan_rows.size > 0:
    #     df_sel = df_sel.drop(df.index[nan_rows])
    # g = sns.PairGrid(df_sel, diag_sharey=False)
    # g.map_lower(sns.kdeplot, cmap="Blues_d", shade=True, shade_lowest=False)
    # g.map_upper(plt.scatter, s=1)
    # g.map_diag(sns.kdeplot, lw=2)
    #
    # df_sel = df.get(['global_score', 'vr_score', 'spkchanvar_score', 'channel_with_spikes_ratio', 'n_spikes_mean',
    #                  'fmin'])
    # g = sns.PairGrid(df, vars=['global_score', 'vr_score', 'spkchanvar_score'],
    #                  hue='lif_tau_coeff')
    # g = g.map_diag(plt.hist)
    # g = g.map_offdiag(plt.scatter, s=1)
    # g = g.add_legend()
    #
    # grp = df.groupby(['fmin', 'fmax', 'n_channels', 'freq_scale', 'fbank_type', 'forder', 'comp_factor', 'comp_gain',
    #                   'lif_tau_coeff', 'lif_v_thresh_start', 'lif_v_thresh_stop']).indices
    # grp_index = list(grp.values())


