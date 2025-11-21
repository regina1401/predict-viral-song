import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import io

# Page setup
st.set_page_config(page_title="🎶 Viral Music Data Analysis", layout="wide")

st.title("🎶 Viral Music Dataset Analysis")
st.write("Upload a CSV file to explore music features interactively.")

# --- File Upload ---
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file:
    # Load dataset
    data = pd.read_csv(uploaded_file)

    st.subheader("📂 Dataset Preview")
    st.dataframe(data.head())

    # Data Summary
    st.subheader("📊 Data Summary")
    st.write("**Shape:**", data.shape)
    st.write("**Columns:**", list(data.columns))
    st.write("**Statistical Summary:**")
    st.dataframe(data.describe(include="all"))

    # --------------------- Interactive Filtering ---------------------
    st.sidebar.header("🎛️ Interactive Filters")

    # Song search bar
    search_term = st.sidebar.text_input("🔍 Search by Song Name (partial match)")

    # Tempo filter
    tempo_min, tempo_max = st.sidebar.slider(
        "Tempo (BPM) Range",
        min_value=float(data["tempo"].min()),
        max_value=float(data["tempo"].max()),
        value=(float(data["tempo"].min()), float(data["tempo"].max()))
    )

    # Duration filter
    duration_min, duration_max = st.sidebar.slider(
        "Duration (seconds) Range",
        min_value=float(data["duration"].min()),
        max_value=float(data["duration"].max()),
        value=(float(data["duration"].min()), float(data["duration"].max()))
    )

    # Key filter
    key_options = st.sidebar.multiselect(
        "Select Musical Keys",
        options=sorted(data["key"].unique()),
        default=sorted(data["key"].unique())
    )

    # Mode filter
    mode_options = st.sidebar.multiselect(
        "Select Modes",
        options=sorted(data["mode"].unique()),
        default=sorted(data["mode"].unique())
    )

    # Apply filters
    filtered_data = data[
        (data["tempo"].between(tempo_min, tempo_max))
        & (data["duration"].between(duration_min, duration_max))
        & (data["key"].isin(key_options))
        & (data["mode"].isin(mode_options))
    ]

    # Apply search filter
    if search_term:
        filtered_data = filtered_data[filtered_data["song_name"].str.contains(search_term, case=False, na=False)]

    st.subheader("🎯 Filtered Data")
    st.write(f"Showing **{len(filtered_data)}** songs after applying filters.")
    st.dataframe(filtered_data)

    # --------------------- Visualizations ---------------------
    st.sidebar.header("📊 Visualization Options")

    if st.sidebar.checkbox("Show Correlation Heatmap", True):
        st.subheader("🔗 Correlation Heatmap")
        if len(filtered_data) > 1:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(
                filtered_data.select_dtypes(include="float64").corr(),
                annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, ax=ax
            )
            st.pyplot(fig)
        else:
            st.info("Need at least 2 rows for correlation heatmap.")

    if st.sidebar.checkbox("Show Tempo Distribution", True):
        st.subheader("⏩ Tempo Distribution")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(filtered_data["tempo"], bins=10, kde=True, color="skyblue", ax=ax)
        ax.set_xlabel("Tempo (BPM)")
        ax.set_ylabel("Count")
        st.pyplot(fig)

    if st.sidebar.checkbox("Show Danceability vs Tempo Scatter", True):
        st.subheader("💃 Danceability vs Tempo")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.scatterplot(
            x="tempo", y="danceability", hue="mode",
            data=filtered_data, s=100, palette="Set2", ax=ax
        )
        ax.set_xlabel("Tempo (BPM)")
        ax.set_ylabel("Danceability")
        st.pyplot(fig)

    if st.sidebar.checkbox("Show Key Distribution", True):
        st.subheader("🎼 Key Distribution by Mode")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.countplot(x="key", hue="mode", data=filtered_data, palette="viridis", ax=ax)
        ax.set_xlabel("Musical Key")
        ax.set_ylabel("Count")
        st.pyplot(fig)

    # --------------------- Clustering ---------------------
    st.sidebar.subheader("🤖 Clustering (K-Means)")
    if st.sidebar.checkbox("Perform Clustering"):
        st.subheader("🎯 Song Clustering")
        features = st.multiselect(
            "Select Features for Clustering",
            options=list(filtered_data.select_dtypes(include="float64").columns),
            default=["tempo", "danceability", "loudness"]
        )
        n_clusters = st.slider("Number of Clusters", 2, 6, 3)

        if features and len(filtered_data) > 0:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(filtered_data[features])
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = kmeans.fit_predict(X_scaled)

            filtered_data["Cluster"] = clusters
            st.write(f"Clustering completed with **{n_clusters} clusters**.")
            st.dataframe(filtered_data[["song_name"] + features + ["Cluster"]])

            if len(features) >= 2:
                st.subheader("Cluster Visualization")
                fig, ax = plt.subplots(figsize=(8, 5))
                sns.scatterplot(
                    x=features[0], y=features[1], hue="Cluster",
                    palette="Set1", data=filtered_data, s=100, ax=ax
                )
                ax.set_title("Cluster Plot (first two features)")
                st.pyplot(fig)

    # --------------------- Download ---------------------
    st.sidebar.subheader("⬇️ Download Options")
    if st.sidebar.checkbox("Enable Download"):
        st.subheader("⬇️ Download Processed Data")
        buffer = io.StringIO()
        filtered_data.to_csv(buffer, index=False)
        st.download_button(
            label="Download CSV",
            data=buffer.getvalue(),
            file_name="filtered_music_data.csv",
            mime="text/csv"
        )

else:
    st.info("👆 Upload a CSV file to start the analysis.")
