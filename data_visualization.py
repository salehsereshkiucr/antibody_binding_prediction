import numpy as np
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
import sklearn.tree as tree


def tsne_vis(X_, Y_, transform='pca'):
    X__ = np.reshape(X_, (X_.shape[0], X_.shape[1] * X_.shape[2]))
    Y__ = np.argmax(Y_, axis=1)
    if transform == 'tsne':
        X_embedded = TSNE(n_components=2).fit_transform(X__)
    if transform == 'pca':
        X_embedded = PCA(n_components=2).fit_transform(X__)
    plt.figure(figsize=(6, 5))
    plt.scatter(X_embedded[Y__ == 0, 0], X_embedded[Y__ == 0, 1], c='r', label='non_binder')
    plt.scatter(X_embedded[Y__ == 1, 0], X_embedded[Y__ == 1, 1], c='b', label='binder')
    plt.legend()
    plt.savefig(transform+'.png')
    plt.clf()
    plt.close()

np.reshape(x_train, (x_train.shape[0], x_train.shape[1] * x_train.shape[2]))
np.reshape(x_test, (x_test.shape[0], x_test.shape[1] * x_test.shape[2]))

def tree_visualization(model):
    fig = plt.figure(figsize=(40,40))
    plt.tight_layout()
    _ = tree.plot_tree(model, fontsize=2, filled=False)
    fig.savefig("decistion_tree.png", dpi=400)

