import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

import tsne_algorithm.tsne


def visualise_tsne_data(X, y, title):
    numbers = y.label.unique()
    plt.figure(figsize=(8, 8))
    for number in numbers:
        plt.scatter(X[y_sample.label == number, 0], X[y_sample.label == number, 1])
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title(title)
    plt.legend(numbers)
    plt.show()


# load MNIST dataset
X = pd.read_csv(r"../dataset/MNIST/mnist_train.csv", header=None)
y = pd.read_csv(r"../dataset/MNIST/train_labels.csv", names=['label'])
y = y - 1

# show random sample of 64 images
random_sample_indices = y.sample(64).index

X_sample = X.iloc[random_sample_indices, :]
y_sample = y.iloc[random_sample_indices, :]

# dataset visualization
fig = plt.figure(figsize=(8, 8))
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
for i in range(len(random_sample_indices)):
    plt.subplot(8, 8, i + 1)
    plt.imshow(X_sample.iloc[i, :].values.reshape(28, -1), cmap=plt.cm.gray, interpolation='nearest')
    plt.xticks([]), plt.yticks([])

plt.show()

##################################
# run t-SNE on 4000 random samples
random_sample_indices = y.sample(4000).index
X_sample = X.iloc[random_sample_indices, :]
y_sample = y.iloc[random_sample_indices, :]

# reduce input data to 30 dimension by using PCA
pca = PCA(n_components=30)
X_pca = pca.fit_transform(X_sample.values)

# run scikit-learn exact t-SNE method
tsne_sl = TSNE(n_components=2, perplexity=40, method="exact", early_exaggeration=4.0, random_state=1, verbose=1)
X_tsne_sl = tsne_sl.fit_transform(X_pca)

# tsne visualisation
visualise_tsne_data(X_tsne_sl, y_sample, 'scikit-learn TSNE on MNIST data')

# run implemented exact t-SNE method
X_tsne = tsne_algorithm.tsne.TSNE(X_pca, perplexity=40, n_iter=1000, early_exaggeration=4.0, method="exact",
                                  random_state=1, verbose=1)

# tsne visualisation
visualise_tsne_data(X_tsne, y_sample, 'implemented TSNE on MNIST data')

###########################################
# run Barnes-Hut-SNE on 1000 random samples
random_sample_indices = y.sample(1000).index
X_sample = X.iloc[random_sample_indices, :]
y_sample = y.iloc[random_sample_indices, :]

# reduce input data to 30 dimension by using PCA
pca = PCA(n_components=30)
X_pca = pca.fit_transform(X_sample.values)

# run scikit-learn Barnes-Hut-SNE method
tsne_sl_bh = TSNE(n_components=2, perplexity=40, method="barnes_hut", early_exaggeration=12.0, random_state=1,
                  verbose=1)
X_tsne_bh_sl = tsne_sl_bh.fit_transform(X_pca)

# tsne visualisation
visualise_tsne_data(X_tsne_bh_sl, y_sample, 'scikit-learn Barnes-Hut-SNE on MNIST data')

# run implemented Barnes-Hut-SNE method
X_tsne_bh = tsne_algorithm.tsne.TSNE(X_pca, perplexity=40, n_iter=1000, early_exaggeration=12.0, method="barnes_hut",
                                     random_state=1, verbose=1)

# tsne visualisation
visualise_tsne_data(X_tsne_bh_sl, y_sample, 'implemented Barnes-Hut-SNE on MNIST data')
