import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import plotly.express as px

# Page Config

st.set_page_config(page_title="Customer Segmentation Dashboard", layout="wide")

# Title

st.title("Customer Segmentation Dashboard")
st.markdown("Analyze and visualize customer clusters using *K-Means Clustering*.")

# Upload Dataset

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Dataset Preview")
    st.dataframe(df, use_container_width=True)

    # Data Preprocessing

    st.subheader("Data Preprocessing")
    numeric_df = df.select_dtypes(include=['int64', 'float64'])
    st.write("Selected Numeric Features for Clustering:")
    st.write(numeric_df.columns.tolist())

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(numeric_df)

    # K-Means Clustering
    
    k = st.slider("Select number of clusters (k)", 2, 10, 3)
    kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
    df["Cluster"] = kmeans.fit_predict(scaled_data)

    st.subheader("Clustered Data")
    st.dataframe(df, use_container_width=True)

    # Visualization

    st.subheader("Cluster Visualization (Seaborn)")

    if numeric_df.shape[1] >= 2:
        fig, ax = plt.subplots(figsize=(10, 6))  
        sns.scatterplot(
            x=numeric_df.iloc[:, 0],
            y=numeric_df.iloc[:, 1],
            hue=df["Cluster"],
            palette="Set1",
            ax=ax
        )
        ax.set_title("Customer Segments")
        st.pyplot(fig)

    # Visualization

    st.subheader("Interactive Cluster Visualization (Plotly)")

    if numeric_df.shape[1] >= 2:
        fig = px.scatter(
            df,
            x=numeric_df.columns[0],
            y=numeric_df.columns[1],
            color="Cluster",
            title="Customer Segments (Interactive)",
            template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)

    # Cluster Count

    st.subheader("Cluster Counts")
    cluster_counts = df["Cluster"].value_counts().reset_index()
    cluster_counts.columns = ["Cluster", "Count"]

    fig = px.bar(cluster_counts, x="Cluster", y="Count", title="Number of Customers per Cluster")
    st.plotly_chart(fig, use_container_width=True)

else:
    st.info("Upload a CSV file to start analysis.")
