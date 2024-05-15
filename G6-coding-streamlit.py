import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, AffinityPropagation
from minisom import MiniSom
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.model_selection import ParameterGrid
from sklearn.impute import SimpleImputer

# Load data
@st.cache_data
def load_data():
    data = pd.read_csv("us_tornado_dataset_1950_2021_streamlit.csv")
    return data

# Dimensionality reduction
def dimensionality_reduction(X, method, n_components):
    if method == 't-SNE':
        # Check if n_components is less than 4 for Barnes-Hut algorithm
        if n_components >= 4:
            raise ValueError("'n_components' should be inferior to 4 for the barnes_hut algorithm.")
        reducer = TSNE(n_components=n_components, method='barnes_hut')
        X_reduced = reducer.fit_transform(X)
    elif method == 'PCA':
        reducer = PCA(n_components=n_components)
        X_reduced = reducer.fit_transform(X)
    else:
        raise ValueError("Unsupported dimensionality reduction method. Choose 't-SNE' or 'PCA'.")
    return X_reduced

# Perform data preprocessing to handle missing values
def preprocess_data(X):
    # Impute missing values using mean strategy
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)
    return X_imputed

# Perform KMeans clustering
def perform_kmeans_clustering(X_reduced, n_clusters):
    model = KMeans(n_clusters=n_clusters)
    cluster_labels = model.fit_predict(X_reduced)
    return cluster_labels

# Perform DBSCAN clustering
def perform_dbscan_clustering(X_reduced, eps, min_samples):
    model = DBSCAN(eps=eps, min_samples=min_samples)
    cluster_labels = model.fit_predict(X_reduced)
    return cluster_labels

# Perform Hierarchical clustering
def perform_hierarchical_clustering(X_reduced, n_clusters):
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    cluster_labels = model.fit_predict(X_reduced)
    return cluster_labels

# Perform SOM clustering
def perform_som_clustering(X, grid_size, sigma, learning_rate, num_iterations):
    # Convert DataFrame to NumPy array
    X_array = X.values
    som = MiniSom(grid_size[0], grid_size[1], X_array.shape[1], sigma=sigma, learning_rate=learning_rate)
    som.train_batch(X_array, num_iterations)
    cluster_labels = np.array([som.winner(x) for x in X_array])
    return cluster_labels

# Perform Affinity Propagation clustering
def perform_affinity_propagation_clustering(X_reduced, damping):
    model = AffinityPropagation(damping=damping)
    cluster_labels = model.fit_predict(X_reduced)
    return cluster_labels

# Plot clustering results
def plot_clusters(X_reduced, cluster_labels, method):
    data = []
    for cluster in np.unique(cluster_labels):
        cluster_indices = np.where(cluster_labels == cluster)[0]
        cluster_data_x = X_reduced[cluster_indices, 0]
        cluster_data_y = X_reduced[cluster_indices, 1]
        
        cluster_data = go.Scatter(
            x=cluster_data_x,
            y=cluster_data_y,
            mode='markers',
            name=f'Cluster {cluster}'
        )
        data.append(cluster_data)

    layout = go.Layout(
        title=f'{method} Clustering Results',
        xaxis=dict(title='Dimension 1'),
        yaxis=dict(title='Demension 2'),
        showlegend=True
    )

    fig = go.Figure(data=data, layout=layout)
    st.plotly_chart(fig)

def plot_cluster_size_distribution(cluster_labels):
    cluster_counts = pd.Series(cluster_labels).value_counts()
    plt.bar(cluster_counts.index, cluster_counts.values)
    plt.xlabel('Cluster Label')
    plt.ylabel('Number of Data Points')
    plt.title('Cluster Size Distribution')
    st.pyplot()

st.set_option('deprecation.showPyplotGlobalUse', False)

def plot_cluster_centroids(X_reduced, cluster_labels, n_clusters):
    centroids = []
    for cluster in range(n_clusters):
        cluster_indices = np.where(cluster_labels == cluster)[0]
        cluster_centroid = np.mean(X_reduced[cluster_indices], axis=0)
        centroids.append(cluster_centroid)

    centroids = np.array(centroids)
    plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', color='red', label='Centroids')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.title('Cluster Centroids')
    plt.legend()
    st.pyplot()

# Joint plot using KDE plot and Regression Plot
def plot_joint_plot(X, cluster_labels):
    # Ensure X and cluster_labels have the same length
    if len(X) != len(cluster_labels):
        st.error("Data and cluster labels have different lengths.")
        return

    # Ensure X has at least 2 dimensions
    if X.shape[1] < 2:
        st.error("Input data must have at least 2 dimensions for joint plotting.")
        return

    # Create DataFrame with dimensions and cluster labels
    data = pd.DataFrame({'Dimension 1': X[:, 0], 'Dimension 2': X[:, 1], 'Cluster': cluster_labels})

    # Plot joint plot
    joint_plot = sns.jointplot(x="Dimension 1", y="Dimension 2", hue="Cluster", kind="kde", data=data)
    st.pyplot()


# Main function to define Streamlit app
def main():
    st.title('Tornado Clustering Analysis')

    # Load data
    data = load_data()

    # Preprocess data to handle missing values
    X = data.drop(columns=['yr', 'mo', 'dy', 'date', 'st'])
    X_imputed = preprocess_data(X)

    # Choose dimensionality reduction method
    dim_reduction_method = st.selectbox('Choose dimensionality reduction method:', ['t-SNE', 'PCA'])

    # Choose number of components for dimensionality reduction
    n_components = st.slider('Select number of components:', min_value=2, max_value=10, value=2)

    # Perform dimensionality reduction
    X_reduced = dimensionality_reduction(X_imputed, dim_reduction_method, n_components)

    # Initialize damping with a default value
    damping = None

    # Choose clustering method
    clustering_method = st.selectbox('Choose clustering method:', ['KMeans', 'DBSCAN', 'Hierarchical', 'SOM', 'Affinity Propagation'])

    # Choose number of clusters or other parameters
    if clustering_method == 'KMeans' or clustering_method == 'Hierarchical' or clustering_method == 'Affinity Propagation':
        n_clusters = st.slider('Select number of clusters:', min_value=2, max_value=10, value=2)
    elif clustering_method == 'DBSCAN':
        eps = st.slider('Select eps:', min_value=0.1, max_value=1.0, value=0.5)
        min_samples = st.slider('Select min_samples:', min_value=2, max_value=10, value=2)
    elif clustering_method == 'SOM':
        grid_size = (st.slider('Select grid size - rows:', min_value=1, max_value=10, value=5),
                     st.slider('Select grid size - columns:', min_value=1, max_value=10, value=5))
        sigma = st.slider('Select sigma:', min_value=0.1, max_value=1.0, value=0.5)
        learning_rate = st.slider('Select learning rate:', min_value=0.1, max_value=1.0, value=0.5)
        num_iterations = st.slider('Select number of iterations:', min_value=100, max_value=1000, value=500)

    # Perform clustering
    if clustering_method == 'KMeans':
        cluster_labels = perform_kmeans_clustering(X_reduced, n_clusters)
    elif clustering_method == 'DBSCAN':
        cluster_labels = perform_dbscan_clustering(X_reduced, eps, min_samples)
    elif clustering_method == 'Hierarchical':
        cluster_labels = perform_hierarchical_clustering(X_reduced, n_clusters)
    elif clustering_method == 'SOM':
        cluster_labels = perform_som_clustering(X, grid_size, sigma, learning_rate, num_iterations)
    elif clustering_method == 'Affinity Propagation':
        damping = st.slider('Select damping:', min_value=0.5, max_value=1.0, value=0.9) # Add damping slider
        cluster_labels = perform_affinity_propagation_clustering(X_reduced, damping)

    # Plot clustering results
    plot_clusters(X_reduced, cluster_labels, clustering_method)

    # Additional plots
    if clustering_method == 'KMeans':
        plot_cluster_size_distribution(cluster_labels)
        plot_cluster_centroids(X_reduced, cluster_labels, n_clusters)

    # Plot joint plot for clustering methods other than SOM
    if clustering_method != 'SOM':
        plot_joint_plot(X_reduced, cluster_labels)

if __name__ == '__main__':
    main()