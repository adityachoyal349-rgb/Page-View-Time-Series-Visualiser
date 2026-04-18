# Page-View-Time-Series-Visualiser

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1. Import data
df = pd.read_csv('medical_examination.csv')

# 2. Overweight column (BMI > 25 → 1, else 0)
df['overweight'] = (df['weight'] / (df['height'] / 100) ** 2 > 25).astype(int)

# 3. Normalize cholesterol and gluc (1 → 0, >1 → 1)
df['cholesterol'] = (df['cholesterol'] > 1).astype(int)
df['gluc']        = (df['gluc']        > 1).astype(int)

def draw_cat_plot():
    # 4. Melt into long format
    df_cat = pd.melt(
        df,
        id_vars='cardio',
        value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight']
    )

    # 5. Group by cardio, variable, value — get counts, rename for catplot
    df_cat = (
        df_cat
        .groupby(['cardio', 'variable', 'value'])
        .size()
        .reset_index(name='total')   # catplot needs a column called 'total'
    )

    # 6. Draw catplot
    fig = sns.catplot(
        data=df_cat,
        x='variable', y='total',
        hue='value',
        col='cardio',
        kind='bar'
    ).fig

    # 7. Save & return
    fig.savefig('catplot.png')
    return fig

def draw_heat_map():
    # 8. Clean data
    df_heat = df[
        (df['ap_lo']  <= df['ap_hi'])                          &
        (df['height'] >= df['height'].quantile(0.025))         &
        (df['height'] <= df['height'].quantile(0.975))         &
        (df['weight'] >= df['weight'].quantile(0.025))         &
        (df['weight'] <= df['weight'].quantile(0.975))
    ]

    # 9. Correlation matrix
    corr = df_heat.corr()

    # 10. Upper-triangle mask
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # 11. Figure
    fig, ax = plt.subplots(figsize=(12, 9))

    # 12. Heatmap
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt='.1f',
        center=0,
        vmin=-0.1,
        vmax=0.3,
        square=True,
        linewidths=0.5,
        cbar_kws={'shrink': 0.5},
        ax=ax
    )

    # 13. Save & return
    fig.savefig('heatmap.png')
    return fig
