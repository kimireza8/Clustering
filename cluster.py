import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

dataset = pd.read_csv('heart.csv')  

data = dataset.loc[:, ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']]
train_label = dataset['target']

print(data)
print(train_label)


sc = MinMaxScaler(feature_range=(0, 1))
train_data_scaled = sc.fit_transform(data)
print("Normalisasi heart : ", train_data_scaled)

clustering = KMeans(n_clusters=2, init="random", n_init=1)
clusters = clustering.fit_predict(train_data_scaled)
print("Hasil cluster K-means : ", clusters)

clusteringHC1 = AgglomerativeClustering(n_clusters=2, linkage='single')
clustersHC1 = clusteringHC1.fit_predict(train_data_scaled)
print('\nHasil clustering single:\n', clustersHC1)

clusteringHC2 = AgglomerativeClustering(n_clusters=2, linkage='average')
clustersHC2 = clusteringHC2.fit_predict(train_data_scaled)
print('\nHasil clustering average:\n', clustersHC2)

clusteringHC3 = AgglomerativeClustering(n_clusters=2, linkage='complete')
clustersHC3 = clusteringHC3.fit_predict(train_data_scaled)
print('\nHasil clustering complete:\n', clustersHC3)

chi2_selector = SelectKBest(chi2, k=3)
chi2_selector.fit(data, train_label)

cols = chi2_selector.get_support(indices=True)
df_selected_features = data.iloc[:, cols]
print(df_selected_features.head())

sse_list = []
clusters_list = []

for i in range(10):
    kmeans = KMeans(n_clusters=3, init='random', n_init=10, random_state=i)
    clusters = kmeans.fit_predict(df_selected_features)
    sse = kmeans.inertia_
    sse_list.append(sse)
    clusters_list.append(clusters)
    print(f"\nIteration {i+1}, SSE: {sse}")
    print(clusters)

min_sse = min(sse_list)
min_sse_indices = [index for index, sse in enumerate(sse_list) if sse == min_sse]

print(f"\nMinimum SSE: {min_sse}")

for index in min_sse_indices:
    print(f"\nBest Clustering (k=3) from iteration {index+1} with SSE: {min_sse}")
    print(clusters_list[index])