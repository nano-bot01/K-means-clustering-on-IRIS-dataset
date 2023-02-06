# K-means-clustering-on-IRIS-dataset
K means clustering on IRIS dataset on Sapel length and Sapel width


### Importing the Dependencies 

```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
```

### Import data from dataset

```
dataset = pd.read_csv("C://Users//DITU//Documents//Jupyter//iris_dataset.csv")
```

### Take two attributes for training model with K-Means clustering

> Here, I have taken Sepal Length and Sepal Width as X and Y attributes respectively.

### Zip the dataset into one list 

```
dataset_new = list(zip(X,Y))
dataset_new

```

### Train model 
```
iris_model = KMeans(n_clusters = 4)
iris_model.fit(dataset_new)
```


### Visualizing the clusters

```
plt.figure(figsize=(16,9))
plt.scatter(X,Y, c= iris_model.labels_, marker= "*", s = 300)
plt.xlabel("Iris Sepal Length", fontsize = 12)
plt.ylabel("Iris Sepal Width", fontsize = 12)
plt.title("K Means cluster", fontsize = 16)

plt.show()
```

## Please Upvote and follow for more ☝️☝️☝️

### [Ankit Nainwal](https://github.com/nano-bot01)



