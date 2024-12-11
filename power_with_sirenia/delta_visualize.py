''' 
visualize normilized power for each frequency
'''

import argparse
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--delta_df_dir", type=str, required = True,
                    help = 'copy paste directory for delta df')
parser.add_argument("--save_dir", type = str, default='save_imgs/', help = 'directory to save output figures')
args = parser.parse_args()
df_dir = args.delta_df_dir
save_dir = args.save_dir

df = pd.read_csv(f'{df_dir}')
for freq in ['delta', 'theta', 'alpha', 'beta', 'gamma']:
    df[freq] = df[freq].astype(complex)

convert_list = []
for index, row in df.iterrows():
    delta = row['delta']
    theta = row['theta']
    alpha = row['alpha']
    beta = row['beta']
    gamma = row['gamma']
    convert_list.append([row['group'], row['ID'], row['phase'], row['state'], 'delta', delta.real])
    convert_list.append([row['group'], row['ID'], row['phase'], row['state'], 'theta', theta.real])
    convert_list.append([row['group'], row['ID'], row['phase'], row['state'], 'alpha', alpha.real])
    convert_list.append([row['group'], row['ID'], row['phase'], row['state'], 'beta', beta.real])
    convert_list.append([row['group'], row['ID'], row['phase'], row['state'], 'gamma', gamma.real])

convert_df = pd.DataFrame(convert_list, columns=['group', 'ID', 'phase', 'state', 'freq', 'value'])
convert_df.to_csv('./delta_df2.csv', index=False) 

def plot_conditions(phase, state):
    df1 = convert_df.loc[(convert_df["phase"] == phase) & (convert_df['state'] == state)]
    _max = df1['value'].max()
    fig, ax = plt.subplots()
    #ax.set_ylim(0, _max+10)
    ax.set_ylim(0, 60)
    pal = ['#91cb98', '#afdee8']
    sns.barplot(df1, x='freq', y='value', hue = 'group', ax = ax, palette = pal)
    sns.stripplot(x='freq',
        y='value', hue = 'group', legend=None,
        data=df1, dodge=True, alpha=0.7, ax= ax , palette= ['black', 'black'], s = 10, linewidth = 0.1, marker="$\circ$")
    ax.set(xlabel=None)
    ax.set(ylabel = 'Normalized Power')
    capital_phase = phase.capitalize()
    if state == 'nrem':
        capital_state = state.upper()
    elif state == 'rem':
        capital_state = state.upper()
    else:
        capital_state = state.capitalize()

    ax.title.set_text(f'{capital_state} ({capital_phase} Phase)')
    plt.savefig(f'{save_dir}normalized power_{phase}_{state}.png', dpi=fig.dpi)
    plt.show()

for phase in ['dark', 'light']:
    for state in ['wake', 'nrem', 'rem']:
        plot_conditions(phase, state)
