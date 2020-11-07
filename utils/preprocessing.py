import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot_bar(df,var):
    labs = df[var].value_counts()
    ax = labs.plot(kind='bar',title="Nombre d'observations par catégorie")
    ax.set_xlabel("Catégorie")
    ax.set_ylabel("Nombre")
    plt.show()

def var_density(data,var,group):
    data.groupby(group)[var].plot(kind='kde',alpha=0.7,legend=True)

def plot_most_pos_target(data,n):
    temp = data.apply(pd.value_counts).iloc[:2].drop(['sig_id'], axis=1).reset_index()
    temp2 = temp.sort_values(temp.last_valid_index(), axis=1).iloc[1,-n:]
    temp2.plot(kind='bar')
