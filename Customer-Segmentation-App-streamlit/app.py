import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
from io import BytesIO
from sklearn.cluster import KMeans

#  PAGE CONFIGURATION
st.set_page_config(page_title="Dashboard", layout="wide")

st.title("Dashboard")
st.markdown("### Analyze customer behavior and visualize clusters interactively.")
st.write("Upload your dataset (CSV) below or explore the sample dataset preview.")
st.write("CSV Format(download and test)")
#  SAMPLE DATA
sample_data = {
    "CustomerID": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
    "Gender": ["Male", "Male", "Female", "Female", "Female", "Female", "Female", "Female", "Male", "Female", 
               "Male", "Female", "Female", "Female", "Male", "Male", "Female", "Male", "Male", "Female"],
    "Age": [19, 21, 20, 23, 31, 22, 35, 23, 64, 30, 67, 35, 58, 24, 37, 22, 35, 20, 52, 35],
    "Annual Income (k$)": [15, 15, 16, 16, 17, 17, 18, 18, 19, 19, 19, 19, 20, 20, 20, 20, 21, 21, 23, 23],
    "Spending Score (1-100)": [39, 81, 6, 77, 40, 76, 6, 94, 3, 72, 14, 99, 15, 77, 13, 79, 35, 66, 29, 98]
}
st.dataframe(pd.DataFrame(sample_data))

#  FILE UPLOAD
uploaded_file = st.file_uploader("Upload your customer dataset (CSV)", type=["csv"])
if uploaded_file:
    data = pd.read_csv(uploaded_file)
else:
    data = pd.DataFrame(sample_data)

st.markdown("### Sample Dataset Preview")
st.dataframe(data.head())

#  DOWNLOAD FUNCTION
def download_plotly(fig, filename):
    buffer = BytesIO()
    fig.write_image(buffer, format="png")
    st.download_button(
        label=f"Download {filename}",
        data=buffer.getvalue(),
        file_name=filename,
        mime="image/png"
    )

#  TABS
tabs = st.tabs([
    "Gender Distribution", "Age Distribution", "Annual Income",
     "Elbow Method", "Clusters"
])

#  GENDER DISTRIBUTION
with tabs[0]:
    fig = px.histogram(data, x="Gender", color="Gender", title="Gender Distribution",
                       color_discrete_sequence=["#5DADE2", "#2874A6"])
    fig.update_layout(template="simple_white", title_x=0.5)
    st.plotly_chart(fig, use_container_width=True)
    download_plotly(fig, "gender_distribution.png")

#  AGE DISTRIBUTION
with tabs[1]:
    fig = px.histogram(data, x="Age", nbins=10, title="Age Distribution",
                       color_discrete_sequence=["#5DADE2"])
    fig.update_layout(template="simple_white", title_x=0.5)
    st.plotly_chart(fig, use_container_width=True)
    download_plotly(fig, "age_distribution.png")

#  ANNUAL INCOME
with tabs[2]:
    fig = px.histogram(data, x="Annual Income (k$)", nbins=10, title="Annual Income Distribution",
                       color_discrete_sequence=["#2E86C1"])
    fig.update_layout(template="simple_white", title_x=0.5)
    st.plotly_chart(fig, use_container_width=True)
    download_plotly(fig, "annual_income_distribution.png")

#  ELBOW METHOD
with tabs[3]:
    X = data[["Annual Income (k$)", "Spending Score (1-100)"]]
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init="k-means++", random_state=42)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)
    fig = px.line(x=range(1, 11), y=wcss, markers=True, title="Elbow Method for Optimal Clusters",
                  labels={"x": "Number of Clusters", "y": "WCSS"},
                  color_discrete_sequence=["#2874A6"])
    fig.update_layout(template="simple_white", title_x=0.5)
    st.plotly_chart(fig, use_container_width=True)
    download_plotly(fig, "elbow_method.png")

#  CLUSTERS
with tabs[4]:
    X = data[["Annual Income (k$)", "Spending Score (1-100)"]]
    kmeans = KMeans(n_clusters=5, init="k-means++", random_state=42)
    data["Cluster"] = kmeans.fit_predict(X)
    fig = px.scatter(data, x="Annual Income (k$)", y="Spending Score (1-100)",
                     color=data["Cluster"].astype(str),
                     title="Customer Segments Visualization",
                     color_discrete_sequence=px.colors.qualitative.Set2)
    fig.update_layout(template="simple_white", title_x=0.5)
    st.plotly_chart(fig, use_container_width=True)
    download_plotly(fig, "cluster_visualization.png")

st.success("Dashboard ready! Switch between tabs to explore all visualizations.")
