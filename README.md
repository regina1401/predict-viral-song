# Predict the Viral Song

This project focuses on exploring the factors that influence a song’s potential to become viral using audio feature extraction and exploratory data analytics. The system utilizes the **Essentia** library to extract detailed musical features and **Streamlit** to build an interactive dashboard for visualization and clustering insights.

---

## 🎯 Project Objective
- Analyze key audio features that may contribute to song virality
- Visualize feature relationships and patterns in musical characteristics
- Group similar songs using clustering techniques

---

## 🛠️ Technologies & Tools Used
- **Python**, **Pandas**, **NumPy**, **Matplotlib**, **Seaborn**
- **Essentia** for audio feature extraction
- **Streamlit** for interactive analytics dashboard
- **Scikit-learn** (K-Means clustering)

---

## 📑 Key Features
- Upload custom song datasets for analysis
- Extract audio attributes: tempo, danceability, loudness, duration, key, mode, etc.
- Visualize correlations and patterns through interactive charts
- Perform **K-Means clustering** to identify groups of musically similar tracks
- Explore how rhythmic and tonal features influence potential virality

---

## 📂 Repository Structure
├── data/ # Audio dataset and CSV files
├── notebooks/ # Analysis notebooks
├── src/ # Core scripts for feature extraction & clustering
├── streamlit_app/ # Streamlit dashboard files
├── results/ # Visual outputs
└── README.md # Documentation


---

## ▶️ How to Run

# Clone the repository
git clone https://github.com/<your-username>/predict-the-viral-song.git

# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
streamlit run app.py
