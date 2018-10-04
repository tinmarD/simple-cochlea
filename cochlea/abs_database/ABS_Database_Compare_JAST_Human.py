import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

res_dir = r'C:\Users\deudon\Desktop\M4\_Results\ABS_Exp_Results'
stim_filename = r'stim_set_#2_carac.csv'
sns.set()
sns.set_context('paper')

# Load csv file with stimuli informations
df = pd.read_csv(os.path.join(res_dir, stim_filename), index_col=0, sep=';')
df_sel = df.loc[:, ['snip_duration', 'n_rep', 'n_dis', 'perf', 'JAST_learning_ratio']]

df_mean_n_rep = df.groupby('n_rep').mean()

df['JAST_learning_ratio_2'] = df['JAST_learning_ratio']*df['prop_target_noise']
df['JAST_active_chan_ratio_2'] = df['JAST_active_chan_ratio']*df['prop_target_noise']

for x_i in ['snip_duration', 'n_dis', 'n_rep']:
    f = plt.figure()
    ax1 = f.add_subplot(121)
    sns.barplot(x=x_i, y='perf', data=df, ax=ax1)
    ax2 = f.add_subplot(122)
    sns.barplot(x=x_i, y='JAST_learning_ratio_2', data=df, ax=ax2)

for x_i in ['snip_duration', 'n_dis', 'n_rep']:
    f = plt.figure()
    ax1 = f.add_subplot(121)
    sns.barplot(x=x_i, y='perf', data=df, ax=ax1)
    ax2 = f.add_subplot(122)
    sns.barplot(x=x_i, y='JAST_active_chan_ratio_2', data=df, ax=ax2)

sns.jointplot(x='perf', y='JAST_learning_ratio', kind="reg", data=df)
sns.jointplot(x='perf', y='JAST_active_chan_ratio', kind="reg", data=df)

plt.figure()
df_perf_group = df.groupby('perf').mean()
df_perf_group.JAST_learning_ratio.plot(kind='bar')

# df_stim.to_csv(os.path.join(res_dir, '{}_out.csv'.format(stim_filename[:-4]), sep=';')


n_rows, n_cols = df.snip_duration.unique().size, df.n_rep.unique().size
for n_dis_i in df.n_dis.unique():
    f = plt.figure()
    i, ymax = 1, df.JAST_learning_ratio.quantile(0.98)
    for i_row, snip_dur_i in enumerate(df.snip_duration.unique()):
        for i_col, n_rep_i in enumerate(df.n_rep.unique()):
            ax = f.add_subplot(n_rows, n_cols, i)
            ax.set_xlim([-0.3, 4.3])
            ax.set_ylim([-3, ymax])
            ax.set_xticks([0, 1, 2, 3, 4])
            i += 1
            row_ind = (df.n_dis == n_dis_i) & (df.snip_duration == snip_dur_i) & (df.n_rep == n_rep_i)
            ax.scatter(df[row_ind].perf, df[row_ind].JAST_learning_ratio)
            # sns.regplot('perf', 'JAST_learning_ratio', truncate=True, data=df[row_ind], ax=ax)
            if i_row == 0 and i_col == 1:
                ax.set(title='N Distractor = {} - Snip. dur. = {}, N Rep = {}'.format(n_dis_i, snip_dur_i, n_rep_i))
            else:
                ax.set(title='Snip. dur. = {}, N Rep = {}'.format(snip_dur_i, n_rep_i))
            if i_col == 0:
                ax.set(ylabel='JAST Learning Ratio')
            if i_row+1 == n_rows:
                ax.set(xlabel='Perf')

