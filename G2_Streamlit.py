import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from io import BytesIO
#import warnings
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import SpectralClustering
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score



@st.cache_data
def load_data():
    return pd.read_csv("tfidf_features_reduced.csv")

def pairplot():
    svd_csv = load_data()  
    st.subheader("Pairplot")
    # Create pairplot
    pairplot = sns.pairplot(svd_csv, diag_kind='kde', markers='.')

    # Convert pairplot to image
    buf_pairplot = BytesIO()
    pairplot.savefig(buf_pairplot, format="png")
    buf_pairplot.seek(0)

    # Display pairplot image in Streamlit
    st.image(buf_pairplot, use_column_width=True, caption="Pairplot of SVD Features")

def heatmap():
    svd_csv = load_data()  
    st.subheader("Heatmap")
    # Create heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(svd_csv.corr(), cmap='viridis', annot=True, fmt=".2f")
    plt.title('Correlation Heatmap of SVD Features')
    
    # Convert heatmap to image
    buf_heatmap = BytesIO()
    plt.savefig(buf_heatmap, format="png")
    buf_heatmap.seek(0)

    # Display heatmap image in Streamlit
    st.image(buf_heatmap, use_column_width=True, caption="Correlation Heatmap of SVD Features")

def scatter_plot():
    svd_csv = load_data()  
    st.subheader("Scatter Plot")
    selected_feature_x = st.selectbox("Select Feature for X-axis", svd_csv.columns)
    selected_feature_y = st.selectbox("Select Feature for Y-axis", svd_csv.columns)

    # Create the scatter plot
    plt.figure(figsize=(8, 5))  # Adjust figure size as desired
    plt.scatter(svd_csv[selected_feature_x], svd_csv[selected_feature_y])
    plt.xlabel(selected_feature_x)
    plt.ylabel(selected_feature_y)
    plt.title("Scatter Plot of SVD Features")
    plt.grid(True)

    # Display the plot in Streamlit
    st.pyplot(plt)
    
        
# Function to perform PCA
def perform_pca(data, n_components):
    pca = PCA(n_components=n_components)
    df_pca = pca.fit_transform(data)
    df_pca = pd.DataFrame(df_pca, columns=[f'P{i}' for i in range(1, n_components + 1)])
    return df_pca    

# Function to perform hierarchical clustering
def hierarchical_clustering(df_pca, n_clusters):
    # Perform Agglomerative hierarchical clustering
    agc = AgglomerativeClustering(n_clusters=n_clusters)
    cluster_labels = agc.fit_predict(df_pca)

    # Create figure object
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot scatter plot with named clusters
    for i in range(n_clusters):  # Loop through each cluster
        cluster_data = df_pca[cluster_labels == i]
        ax.scatter(cluster_data['P1'], cluster_data['P2'], label=f'Cluster {i + 1}')

    ax.set_title("Agglomerative Hierarchical Clusters - Scatter Plot", fontsize=18)
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.legend()

    # Display the plot using Streamlit
    st.pyplot(fig)

    # Compute silhouette score
    silhouette_avg = silhouette_score(df_pca, cluster_labels)

    # Compute Davies-Bouldin index
    davies_bouldin_idx = davies_bouldin_score(df_pca, cluster_labels)

    return silhouette_avg, davies_bouldin_idx


# Function to perform KMeans clustering
def perform_kmeans(data, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(data)
    predicted_clusters_kmeans = kmeans.labels_
    return predicted_clusters_kmeans, kmeans.cluster_centers_

# Function to perform Gaussian Mixture Model clustering
def perform_gmm(data, n_clusters):
    gmm = GaussianMixture(n_components=n_clusters, random_state=42)
    gmm.fit(data)
    predicted_clusters_gmm = gmm.predict(data)
    return predicted_clusters_gmm, gmm.means_

# Function to visualize KMeans clustering
def visualize_kmeans_clusters(data, predicted_clusters_kmeans, cluster_centers):
    fig, ax = plt.subplots(figsize=(8, 6)) 
    for cluster_label in range(len(np.unique(predicted_clusters_kmeans))):
        cluster_points = data[predicted_clusters_kmeans == cluster_label]
        ax.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {cluster_label}', cmap='viridis')
    ax.scatter(cluster_centers[:, 0], cluster_centers[:, 1], marker='x', s=100, color='red', label='Cluster Centers')
    ax.set_title(f'KMeans Clustering with {len(np.unique(predicted_clusters_kmeans))} Clusters')
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.legend()
    st.pyplot(fig)

# Function to visualize Gaussian Mixture Model clustering
def visualize_gmm_clusters(data, predicted_clusters_gmm, cluster_centers):
    fig, ax = plt.subplots(figsize=(8, 6)) 
    for cluster_label in range(len(np.unique(predicted_clusters_gmm))):
        cluster_points = data[predicted_clusters_gmm == cluster_label]
        ax.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {cluster_label}', cmap='viridis')
    ax.scatter(cluster_centers[:, 0], cluster_centers[:, 1], marker='x', s=100, color='red', label='Cluster Centers')  # Add label here
    ax.set_title(f'GMM Clustering with {len(np.unique(predicted_clusters_gmm))} Clusters')
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.legend()
    st.pyplot(fig)

#Function to perform Spectral Clustering
def perform_sc(data, n_cluster, n_neighbor):
        sc = SpectralClustering(n_clusters=n_cluster, affinity='nearest_neighbors', n_neighbors=n_neighbor)
        sc_cluster_label = sc.fit_predict(data)
        return sc_cluster_label

#Function to perform DBSCAN
def perform_dbscan(data, eps, min_samples):
        db = DBSCAN(eps=eps, min_samples=min_samples)
        db_cluster_label = db.fit_predict(data)
        return db_cluster_label


#Function to visualise DBSCAN
def visualize_dbscan_clusters(data, db_cluster_label):
    st.set_option('deprecation.showPyplotGlobalUse', False)
    fig, ax = plt.subplots(figsize=(8, 6)) 
    for cluster_label in range(len(np.unique(db_cluster_label))):
        cluster_points = data[db_cluster_label == cluster_label]
        ax.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {cluster_label}', cmap='viridis')
    ax.set_title(f'DBSCAN Clustering with {len(np.unique(db_cluster_label))} Clusters')
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.legend()
    st.pyplot(fig)
    

            
#Function to visualise Spectral Clustering
def visualize_sc_clusters(data, sc_cluster_label):
    st.set_option('deprecation.showPyplotGlobalUse', False)
    fig, ax = plt.subplots(figsize=(8, 6)) 
    for cluster_label in range(len(np.unique(sc_cluster_label))):
        cluster_points = data[sc_cluster_label == cluster_label]
        ax.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {cluster_label}', cmap='viridis')
    ax.set_title(f'DBSCAN Clustering with {len(np.unique(sc_cluster_label))} Clusters')
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.legend()
    st.pyplot(fig)




# Main function
def main():
    st.sidebar.title('Table of Contents')
    page = st.sidebar.radio('Go to', ['Graph Selection for SVD', 'Hierarchical Clustering', 'Comparative Analysis'])

    if page == 'Graph Selection for SVD':
        st.markdown(
            """
            <style>
                .header {
                    color: #007BFF;
                    text-align: center;
                    font-size: 36px;
                    font-weight: bold;
                    margin-bottom: 20px;
                }
                .description {
                    color: #6C757D;
                    text-align: center;
                    font-size: 18px;
                    margin-bottom: 30px;
                }
                .divider {
                    border-bottom: 2px solid #007BFF;
                    margin-bottom: 30px;
                }
            </style>
            """
            , unsafe_allow_html=True
        )

        st.markdown('<p class="header">Wine Review Dashboard</p>', unsafe_allow_html=True)
        st.markdown('<p class="description">Welcome to Wine Review Dashboard!</p>', unsafe_allow_html=True)
        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        
        st.title("Graph Selection for SVD")
        pairplot_cb = st.checkbox("Show Pairplot")
        scatter_checkbox = st.checkbox("Show Scatter Plot")
        heatmap_cb = st.checkbox("Show Heatmap")
        if pairplot_cb:
            pairplot()
       
        if scatter_checkbox:
            scatter_plot()
           
        if heatmap_cb:
            heatmap()


    elif page == 'Hierarchical Clustering':
        st.title('Hierarchical Clustering')

        # Load data
        data = load_data()  # Define your data loading mechanism

        # Perform PCA
        n_components = 10
        df_pca = perform_pca(data, n_components)
        
        # Slider for selecting number of clusters
        n_clusters = st.sidebar.slider('Select number of clusters', min_value=2, max_value=10, value=4)

        # Perform hierarchical clustering and display results
        silhouette_avg, davies_bouldin_idx = hierarchical_clustering(df_pca[['P1', 'P2']], n_clusters)

        # Display performance metrics
        st.subheader('Clustering Performance Metrics')
        st.write('---')
        st.write(f'**Number of Clusters:** {n_clusters}')
        st.write(f'**Silhouette Score:** {silhouette_avg:.4f}')
        st.write(f'**Davies-Bouldin Index:** {davies_bouldin_idx:.4f}')
        
    elif page == 'Comparative Analysis':
        st.title('Comparative Analysis')
        
        # Load data
        data = load_data()

        st.title('KMeans Clustering')
        # Sidebar for selecting number of clusters
        n_clusters_kmeans = 3
        # Perform KMeans clustering
        predicted_clusters, cluster_centers = perform_kmeans(data, n_clusters_kmeans)
        # Visualize clusters
        visualize_kmeans_clusters(data.values, predicted_clusters, cluster_centers)


        st.title('GMM Clustering')
        # Sidebar for selecting number of clusters
        n_clusters_gmm = 5
        # Perform GMM clustering
        predicted_clusters_gmm, cluster_centers_gmm = perform_gmm(data, n_clusters_gmm)
        # Visualize clusters
        visualize_gmm_clusters(data.values, predicted_clusters_gmm, cluster_centers)

        st.title('Spectral Clustering')
        n_clusters_sc = 2
        n_neighbors_sc = 10
        predicted_clusters_sc = perform_sc(data, n_clusters_sc, n_neighbors_sc)
        visualize_sc_clusters(data.values, predicted_clusters_sc)

        st.title('DBSCAN Clustering')
        eps_db = 0.6
        min_samples_db = 6
        # Perform Spectral clustering
        predicted_clusters_db= perform_dbscan(data, eps_db, min_samples_db)
        # Visualize clusters
        visualize_dbscan_clusters(data.values, predicted_clusters_db)
	


   

if __name__ == "__main__":
    main()