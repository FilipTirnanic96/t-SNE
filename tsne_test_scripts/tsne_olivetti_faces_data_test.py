import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import tsne_algorithm.tsne


def visualise_tsne_data(X, y, title):
    numbers = y.label.unique()
    numbers = [i for i in np.arange(0, 20)]
    number = max(numbers)
    cmap = plt.get_cmap('gnuplot')
    colors = [cmap(i) for i in np.linspace(0, 1, number+1)]
    plt.figure(figsize=(15, 15))
    for i, number in enumerate(numbers):
        plt.scatter(X[y.label == number, 0], X[y.label == number, 1], color=colors[i])
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title(title)
    plt.legend(numbers)
    plt.show()


# Load Olivetti faces data set
X = pd.read_csv(r"../dataset/Olivetti/train_Olivetti.csv", header=None)
y = pd.read_csv(r"../dataset/Olivetti/label_Olivetti.csv", names=['label']).astype(int)

# show random sample of 64 images
random_sample_indices = y.sample(64).index

X_sample = X.iloc[random_sample_indices, :]
y_sample = y.iloc[random_sample_indices, :]

# dataset visualization
fig = plt.figure(figsize= (8, 8))
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
for i in range(len(random_sample_indices)):
    plt.subplot(8, 8, i+1)
    plt.imshow(X_sample.iloc[i, :].values.reshape(64, -1), cmap = plt.cm.gray, interpolation='nearest')
    plt.xticks([]),  plt.yticks([])

plt.show()

##################################
# reduce input data to 30 dimension by using PCA
pca = PCA(n_components=30)
X_pca = pca.fit_transform(X.values)

# run scikit-learn exact t-SNE method
tsne_sl = TSNE(n_components = 2, perplexity = 40, method = "exact", early_exaggeration=4.0, random_state = 1, verbose = 1)
X_tsne_sl = tsne_sl.fit_transform(X_pca)

# tsne visualisation
visualise_tsne_data(X_tsne_sl, y, 'scikit-learn TSNE on Olivetti faces data')

# run implemented exact t-SNE method
X_tsne = tsne_algorithm.tsne.TSNE(X_pca, perplexity = 40, n_iter = 1000, early_exaggeration = 4.0, method = "exact", random_state = 1, verbose = 1)

# tsne visualisation
visualise_tsne_data(X_tsne, y, 'implemented TSNE on Olivetti faces data')

###########################################
# run scikit-learn Barnes-Hut-SNE method
tsne_sl_bh = TSNE(n_components = 2, perplexity = 40, method = "barnes_hut", early_exaggeration=12.0, random_state = 1, verbose = 1)
X_tsne_bh_sl = tsne_sl_bh.fit_transform(X_pca)

# tsne visualisation
visualise_tsne_data(X_tsne_bh_sl, y, 'scikit-learn Barnes-Hut-SNE on Olivetti faces data')

# run implemented Barnes-Hut-SNE method
X_tsne_bh = tsne_algorithm.tsne.TSNE(X_pca, perplexity = 40, n_iter = 1000, early_exaggeration = 12.0, method = "barnes_hut", random_state = 1, verbose = 1)

# tsne visualisation
visualise_tsne_data(X_tsne_bh_sl, y, 'implemented Barnes-Hut-SNE on Olivetti faces data')
