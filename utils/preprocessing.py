import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns


def plot_corr_targets(data):
    f, ax = plt.subplots(figsize=(10, 8))
    corr = data.drop('sig_id',1).corr()
    sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
                square=True, ax=ax)


def plot_corr_mat_features(data,feature_group,var,cat):
    if var != None:
        data = data[data[var] == cat]
    c_cols = [col for col in data.columns if feature_group in col]
    df_cells = data[c_cols]
    f, ax = plt.subplots(figsize=(10, 8))
    corr = df_cells.corr()
    sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool),
            cmap=sns.diverging_palette(220, 10, as_cmap=True),square=True, ax=ax)

def plot_bar(df,var):
    labs = df[var].value_counts()
    ax = labs.plot(kind='bar',title="Nombre d'observations par catégorie")
    ax.set_xlabel("Catégorie")
    ax.set_ylabel("Nombre")
    plt.show()

def multi_var_bar(df,var1,var2):
    group_counts = df.groupby([var1, var2]).sig_id.count().reset_index()
    table = pd.pivot_table(group_counts, index=var1, columns=var2, values='sig_id')
    table.plot(kind='barh', color=['r', 'g', 'b'])

def var_density(data,var,group):
    data.groupby(group)[var].plot(kind='kde',alpha=0.7,legend=True)

def plot_most_pos_target(data,n):
    temp = data.apply(pd.value_counts).iloc[:2].drop(['sig_id'], axis=1).reset_index(drop = True)
    temp2 = temp.sort_values(temp.last_valid_index(), axis=1)
    ordered_cols = temp2.columns
    temp2.iloc[1,-n:].plot(kind='bar')
    return list(ordered_cols)[::-1]

def prepro_X(data1,data2):
    Xt = pd.merge(data1,data2, on='sig_id')
    list_col_to_conv = ['cp_type','cp_dose','cp_time','drug_id']
    for col in list_col_to_conv:
        Xt[col] = pd.Categorical(Xt[col]).codes
    del Xt['sig_id']
    return Xt.to_numpy()

def prepro_Y(data):
    Yt = data.copy()
    del Yt['sig_id']
    return Yt.to_numpy()

def split_and_select_target(features,targets,i):
    if type(i) == int:
        target = targets[:,i]
        x_train, x_val, y_train, y_val = train_test_split(features, target, test_size=0.33, random_state=2020)
    else:
        x_train, x_val, y_train, y_val = train_test_split(features, targets, test_size=0.33, random_state=2020)
    return x_train, x_val, y_train, y_val


def def_positive_rate_per_col(data):
    temp = data.apply(pd.value_counts).iloc[:2].drop(['sig_id'], axis=1).reset_index(drop = True)
    return dict((temp.T[1] / (temp.T[0] + temp.T[1])).T)
