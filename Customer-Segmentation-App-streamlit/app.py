import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Example data
example_data = {
    "CustomerID": [1, 2, 3, 4, 5],
    "Age": [25, 34, 45, 23, 35],
    "Annual Income (k$)": [15, 40, 75, 18, 55],
    "Spending Score (1-100)": [39, 81, 6, 77, 40]
}
example_df = pd.DataFrame(example_data)

st.write("Example CSV format (download and test):")
st.dataframe(example_df)

csv = example_df.to_csv(index=False).encode("utf-8")
st.download_button(
    label=" Download Example CSV",
    data=csv,
    file_name="example_customers.csv",
    mime="text/csv"
)

# Streamlit Page Configuration
st.set_page_config(page_title="Customer Segmentation", layout="wide")

st.title(" Customer Segmentation using KMeans Clustering")

# Upload dataset
uploaded_file = st.file_uploader("Upload your dataset (CSV)", type="csv")

if uploaded_file is not None:
    # Load dataset
    df = pd.read_csv(uploaded_file)
    st.write("### Data Preview", df.head())

    # Select only numerical columns for clustering
    num_df = df.select_dtypes(include=["int64", "float64"])

    if num_df.empty:
        st.error("No numeric columns found for clustering. Please upload a dataset with numeric values.")
    else:
        # Standardize Data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(num_df)

        # Elbow Method
        st.write("### Elbow Method to Find Optimal Clusters")
        distortions = []
        for k in range(1, 11):
            kmeans = KMeans(n_clusters=k, init="k-means++", random_state=42, n_init=10)
            kmeans.fit(scaled_data)
            distortions.append(kmeans.inertia_)

        fig1, ax1 = plt.subplots()
        ax1.plot(range(1, 11), distortions, marker="o")
        ax1.set_xlabel("Number of Clusters (k)")
        ax1.set_ylabel("Distortion / Inertia")
        ax1.set_title("Elbow Method")
        st.pyplot(fig1)

        # User selects cluster count
        n_clusters = st.slider("Select number of clusters", 2, 10, 3)

        # Run KMeans
        kmeans = KMeans(n_clusters=n_clusters, init="k-means++", random_state=42, n_init=10)
        df["Cluster"] = kmeans.fit_predict(scaled_data)

        st.write("### Clustered Data", df.head())

        # Cluster distribution bar chart
        st.write("### Cluster Size Distribution")
        cluster_counts = df["Cluster"].value_counts().sort_index()
        fig2, ax2 = plt.subplots()
        sns.barplot(x=cluster_counts.index, y=cluster_counts.values, ax=ax2, palette="tab10")
        ax2.set_xlabel("Cluster")
        ax2.set_ylabel("Number of Customers")
        ax2.set_title("Cluster Size Distribution")
        st.pyplot(fig2)

        # Scatter plot of first 2 numerical features
        st.write("### Scatter Plot of Clusters (using first 2 numeric features)")
        if num_df.shape[1] >= 2:
            fig3, ax3 = plt.subplots()
            sns.scatterplot(
                x=num_df.iloc[:, 0],
                y=num_df.iloc[:, 1],
                hue=df["Cluster"],
                palette="tab10",
                s=60,
                ax=ax3
            )
            ax3.set_xlabel(num_df.columns[0])
            ax3.set_ylabel(num_df.columns[1])
            ax3.set_title("Customer Clusters")
            st.pyplot(fig3)
        else:
            st.warning("Not enough numeric columns to plot scatter graph.")

        # Download clustered CSV
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Clustered CSV",
            data=csv,
            file_name="customer_clustered.csv",
            mime="text/csv"

        )
