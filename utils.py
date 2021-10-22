"""

Author: Antonio Sutera (sutera.antonio@gmail.com)
License: BSD 3 clause

"""

import numpy as np
import os
from matplotlib import pyplot as plt
import time
import datetime
from contextlib import contextmanager

# Various =====================================================================

@contextmanager
def measure_time(label):
    start = time.time()
    yield
    end = time.time()
    print('Duration of [{}]: {}'.format(label,
                                        datetime.timedelta(seconds=end-start)))

# Visualization ===============================================================

def led_mini(LMDI, SHAP, SAAB, X, idx, save_figure="saved_figures/"):
    # custom color
    from matplotlib.colors import LinearSegmentedColormap
    # get colormap
    ncolors = 256+1
    color_array = plt.get_cmap('RdBu')(range(ncolors))
    mid = int((ncolors-1)/2)
    # create a colormap object
    map_object = LinearSegmentedColormap.from_list(name='red_transparent_blue',colors=color_array)
    plt.register_cmap(cmap=map_object)

    FF = 12
    FF2 = 8
    fig, axs = plt.subplots(1, 4, figsize=(8,3))
    for i in range(4):
        axs[i].spines["top"].set_visible(False)
        axs[i].spines["right"].set_visible(False)
        axs[i].spines["bottom"].set_visible(False)
        axs[i].spines["left"].set_visible(False)
        axs[i].set_xticks([])
        axs[i].set_yticks([])

    A = np.round(0.75, decimals=2) 
    axs[0].imshow(coloring(X[idx,:]),cmap=plt.cm.gray_r)  
    axs[1].imshow(coloring(X[idx,:]),cmap=plt.cm.gray_r,alpha=0.1) 
    axs[2].imshow(coloring(X[idx,:]),cmap=plt.cm.gray_r,alpha=0.1)  
    axs[3].imshow(coloring(X[idx,:]),cmap=plt.cm.gray_r,alpha=0.1)  

    imp_LOCALM = coloring(LMDI[idx,:])
    imp_SAABAS = coloring(SAAB[idx,:])
    imp_SHAP   = coloring(SHAP[idx,:])

    axs[1].imshow(imp_LOCALM,cmap='red_transparent_blue',vmin=-A,vmax=A)  
    axs[2].imshow(imp_SAABAS,cmap='red_transparent_blue',vmin=-A,vmax=A)  
    ff=axs[3].imshow(imp_SHAP,cmap='red_transparent_blue',vmin=-A,vmax=A)  

    axs[0].set_title('Class',fontsize=FF2,rotation=0,loc="left")
    axs[1].set_title('Local MDI',fontsize=FF2,rotation=0,loc="left")
    axs[2].set_title('Saabas',fontsize=FF2,rotation=0,loc="left")
    axs[3].set_title('TreeSHAP',fontsize=FF2,rotation=0,loc="left")
        
    tt = [-A,0,A]
    cbar_ax = fig.add_axes([0.92, 0.125, 0.02, 0.75])
    cb=fig.colorbar(ff, cax=cbar_ax,label="",ticks=tt,orientation="vertical")
    cb.ax.set_yticklabels(['min', '0', 'Max']) 



def digits_mini(LMDI, SHAP, SAAB, X, idx, save_figure="saved_figures/"):
    # custom color
    from matplotlib.colors import LinearSegmentedColormap
    # get colormap
    ncolors = 256+1
    color_array = plt.get_cmap('RdBu')(range(ncolors))
    mid = int((ncolors-1)/2)
    # create a colormap object
    map_object = LinearSegmentedColormap.from_list(name='red_transparent_blue',colors=color_array)
    plt.register_cmap(cmap=map_object)

    EE = [-5.5,5.5,-9,9]
    FF = 12
    FF2 = 8
    fig, axs = plt.subplots(1, 4, figsize=(8,3))
    for i in range(4):
        axs[i].spines["top"].set_visible(False)
        axs[i].spines["right"].set_visible(False)
        axs[i].spines["bottom"].set_visible(False)
        axs[i].spines["left"].set_visible(False)
        axs[i].set_xticks([])
        axs[i].set_yticks([])


    axs[0].imshow(X[idx,:].reshape(8,8),cmap=plt.cm.gray_r,extent=EE)  
    axs[1].imshow(X[idx,:].reshape(8,8),cmap=plt.cm.gray_r,alpha=0.1,extent=EE) 
    axs[2].imshow(X[idx,:].reshape(8,8),cmap=plt.cm.gray_r,alpha=0.1,extent=EE)  
    axs[3].imshow(X[idx,:].reshape(8,8),cmap=plt.cm.gray_r,alpha=0.1,extent=EE)  

    imp_LOCALM = LMDI[idx,:]
    imp_SAABAS = SAAB[idx,:]
    imp_SHAP   = SHAP[idx,:]

    A = 1
    axs[1].imshow(imp_LOCALM.reshape(8,8),cmap='red_transparent_blue',vmin=-A,vmax=A,extent=EE)  
    axs[2].imshow(imp_SAABAS.reshape(8,8),cmap='red_transparent_blue',vmin=-A,vmax=A,extent=EE)  
    ff=axs[3].imshow(imp_SHAP.reshape(8,8),cmap='red_transparent_blue',vmin=-A,vmax=A,extent=EE)  


    axs[0].set_title('Class',fontsize=FF2,rotation=0,loc="left")
    axs[1].set_title('Local MDI',fontsize=FF2,rotation=0,loc="left")
    axs[2].set_title('Saabas',fontsize=FF2,rotation=0,loc="left")
    axs[3].set_title('TreeSHAP',fontsize=FF2,rotation=0,loc="left")

    tt = [-A,0,A]

    cbar_ax = fig.add_axes([0.92, 0.125, 0.02, 0.75])
    cb=fig.colorbar(ff, cax=cbar_ax,label="",ticks=tt,orientation="vertical")
    cb.ax.set_yticklabels(['min', '0', 'Max']) 

# Datasets ====================================================================

def make_led(irrelevant=0):
    """
    Generate exhaustively all samples from the 7-segment problem.

    Parameters
    ----------
    irrelevant : int, optional (default=0)
        The number of irrelevant binary features to add. Since samples are
        generated exhaustively, this makes the size of the resulting dataset
        2^(irrelevant) times larger.

    Returns
    -------
    X, y

    From: https://github.com/glouppe/paper-variable-importances-nips2013/
    """
    data = np.array([[0, 0, 1, 0, 0, 1, 0, 1],
                     [1, 0, 1, 1, 1, 0, 1, 2],
                     [1, 0, 1, 1, 0, 1, 1, 3],
                     [0, 1, 1, 1, 0, 1, 0, 4],
                     [1, 1, 0, 1, 0, 1, 1, 5],
                     [1, 1, 0, 1, 1, 1, 1, 6],
                     [1, 0, 1, 0, 0, 1, 0, 7],
                     [1, 1, 1, 1, 1, 1, 1, 8],
                     [1, 1, 1, 1, 0, 1, 1, 9],
                     [1, 1, 1, 0, 1, 1, 1, 0]])

    X, y = np.array(data[:, :7], dtype=np.bool), data[:, 7]

    if irrelevant > 0:
        X_ = []
        y_ = []

        for i in xrange(10):
            for s in itertools.product(range(2), repeat=irrelevant):
                X_.append(np.concatenate((X[i], s)))
                y_.append(i)

        X = np.array(X_, dtype=np.bool)
        y = np.array(y_)

    return X, y

def make_led_sample(n_samples=200, irrelevant=0, random_state=None):
    """Generate random samples from the 7-segment problem.

    Parameters
    ----------
    n_samples : int, optional (default=200)
        The number of samples to generate.

    irrelevant : int, optional (default=0)
        The number of irrelevant binary features to add.

    Returns
    -------
    X, y

    From: https://github.com/glouppe/paper-variable-importances-nips2013/
    """

    random_state = check_random_state(random_state)

    data = np.array([[0, 0, 1, 0, 0, 1, 0, 1],
                     [1, 0, 1, 1, 1, 0, 1, 2],
                     [1, 0, 1, 1, 0, 1, 1, 3],
                     [0, 1, 1, 1, 0, 1, 0, 4],
                     [1, 1, 0, 1, 0, 1, 1, 5],
                     [1, 1, 0, 1, 1, 1, 1, 6],
                     [1, 0, 1, 0, 0, 1, 0, 7],
                     [1, 1, 1, 1, 1, 1, 1, 8],
                     [1, 1, 1, 1, 0, 1, 1, 9],
                     [1, 1, 1, 0, 1, 1, 1, 0]])

    data = data[random_state.randint(0, 10, n_samples)]
    X, y = np.array(data[:, :7],  dtype=np.bool), data[:, 7]

    if irrelevant > 0:
        X = np.hstack((X, random_state.rand(n_samples, irrelevant) > 0.5))

    return X, y

def dataset_classification(name="led"):
    """ This function returns a dataset given a name 

    Parameters
    ----------
    name : string, optional (default="led")

    Returns
    -------
    X: input
    y: output
    """
    if name =="boston":
        # regression (506x13feat)
        from sklearn.datasets import load_boston
        X, y = load_boston(return_X_y=True)
    elif name == "iris":
        # classification (150x4featx3classes)
        from sklearn.datasets import load_iris
        data = load_iris()
        X = data.data
        y = data.target
    elif name == "diabetes":
        # regression (442x10feat)
        from sklearn.datasets import load_diabetes
        X, y = load_diabetes(return_X_y=True)
    elif name == "digits":
        # classification (1797x64featx10classes)
        from sklearn.datasets import load_digits
        X, y = load_digits(return_X_y=True)
    elif name == "wine":
        # classification (178x13featuresx3classes)
        from sklearn.datasets import load_wine
        X, y = load_wine(return_X_y=True)
    elif name == "breast_cancer":
        # classification (569x30featx2classes)
        from sklearn.datasets import load_breast_cancer
        X, y = load_breast_cancer(return_X_y=True)
    elif name =="nhanesi":
        X,y = shap.datasets.nhanesi()
    elif name == "led":
        X,y = make_led()
    elif name == "led_sampled":
        X,y = make_led_sample()
    else:
        raise ValueError("dataset `{}` not implemented".format(name))
    return X,y


# Others ======================================================================
def coloring(v):
    mat = np.zeros((9,5))
    mat[0,[1,2,3]] = v[0]
    mat[[1,2,3],0] = v[1]
    mat[[1,2,3],4] = v[2]
    mat[4,[1,2,3]] = v[3]
    mat[[5,6,7],0] = v[4]
    mat[[5,6,7],4] = v[5]
    mat[8,[1,2,3]] = v[6]
    return mat