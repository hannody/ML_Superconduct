import numpy as np
from sklearn.decomposition import PCA, NMF

a = np.array((1, 2, 4))  # 4, 5, 6, 7, 8))
#a = np.ones((8, 1))
#a = np.zeros((8, 1))
a = np.reshape(a, (a.shape[0], 1))

print(a)

pca = PCA(n_components=1)
aa = pca.fit_transform(a)
print('-' * 30)
print(aa)
