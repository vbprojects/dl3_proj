#%%
import numpy as np
#%%
npz = np.load("embeddings_nontrain.npz")
X, Y = npz["X"], npz["Y"]
print(X.shape, Y.shape)
#%%
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)
#%%
from sklearn.neighbors import KNeighborsClassifier
# %%
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, Y_train)
#%%
accuracy = knn.score(X_test, Y_test)
# %%
accuracy