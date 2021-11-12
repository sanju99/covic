import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import itertools
import warnings

import os

# import sklearn.impute
# import sklearn.cluster
# import scipy.stats as st
# from sklearn.decomposition import PCA
# from sklearn.cross_decomposition import PLSRegression
# from sklearn.manifold import TSNE

# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import LeaveOneOut, train_test_split

# from sklearn.feature_selection import RFE, RFECV


func_assays = ['ADCP', 'mADCP', 'ADNP', 'mADNP', 'ADCD', 'ADNKA_CD107a', 'ADNKA_MIP-1b']
df_func = pd.read_csv("df_func.csv")

def polar_area_plot(df, col_names, directory=None):
    '''
    Make a polar area (flower) plot for each antibody
    '''
        
    df_split = df[col_names]
    ids = df["CoVIC ID"].values
            
    # Compute pie slices
    N = len(df_split.columns)
    
    #colors = sns.color_palette("colorblind", as_cmap=True)[:N]
    colors = sns.color_palette("deep", as_cmap=True)[:N]
    
    theta = np.linspace(0, 2 * np.pi, N, endpoint=False)
    width = 2 * np.pi / N
        
    # create directory to save files
    if os.path.isdir(directory) == False:
        os.mkdir(directory)

    # make a representative figure with assay titles for the legend
    sns.set_style("dark")
    fig, ax = plt.subplots(figsize=(2, 2), subplot_kw={'projection': 'polar'}, constrained_layout=True)
    
    ax.bar(theta,
          np.ones(N)*np.max(np.max(df_split).values),
          width,
          bottom=0,
          color=colors,
          edgecolor="black")
    

    ax.set_yticklabels([])  
    ax.set_xticks(theta)
    ax.set_xticklabels(col_names)
    ax.tick_params(axis='x', pad=10)

    ax.tick_params(axis='both', which='major', labelsize=10)
    
    fName_legend = os.path.join(directory, "flowers_legend.png")
    plt.savefig(fName_legend)
    
    sns.set_style("dark")

    fig, ax = plt.subplots(13, 12, figsize=(26, 24), subplot_kw={'projection': 'polar'}, constrained_layout=True)

    ax = ax.flatten()
    
    for i in range(len(df)):
    
        radii = df_split.iloc[i, :].values

        ax[i].set_title(f"{ids[i]}", pad=10, fontsize=14)

        ax[i].bar(theta, 
               radii, 
               width=width,
               bottom=0, 
               color=colors,
               edgecolor="black")

        ax[i].set_yticklabels([])
        
        ax[i].set_xticks(theta)
        ax[i].set_xticklabels([])

        ax[i].tick_params(axis='both', which='major', labelsize=10)
        
    fName = os.path.join(directory, "CoVIC_flowers.png")
    plt.savefig(fName)
    
    
polar_area_plot(df_func, func_assays, "figures")