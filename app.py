import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Load data
@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv("metadata.csv", low_memory=False)
    # Convert 'publish_time' into year
    df["year"] = pd.to_datetime(df["publish_time"], errors="coerce").dt.year
    return df

df: pd.DataFrame = load_data()

st.title("CORD-19 Data Exploration")

# Year filter
year_min = int(df["year"].min()) if bool(df["year"].notna().any()) else 2000
year_max = int(df["year"].max()) if bool(df["year"].notna().any()) else 2025

year_range = st.slider("Select Year Range", year_min, year_max, (2015, 2020))

df_filtered: pd.DataFrame = pd.DataFrame(
    df[df["year"].between(year_range[0], year_range[1])]
)

st.write("### Filtered Data Sample")
st.write(df_filtered.head())

# --- Publications per year ---
st.subheader("Publications per Year")
year_counts = df_filtered["year"].value_counts().sort_index()
fig, ax = plt.subplots()
ax.bar(year_counts.index.to_numpy(), year_counts.values.astype(int))
ax.set_xlabel("Year")
ax.set_ylabel("Number of Papers")
st.pyplot(fig)

# --- Top journals ---
st.subheader("Top Journals")
top_journals = df_filtered["journal"].value_counts().head(10)
fig, ax = plt.subplots()
sns.barplot(x=top_journals.values, y=top_journals.index, ax=ax)
ax.set_xlabel("Count")
st.pyplot(fig)

# --- Word Cloud (Titles) ---
st.subheader("Word Cloud of Paper Titles")
titles = " ".join(pd.Series(df_filtered["title"]).dropna().astype(str).tolist())
if titles.strip():
    wc = WordCloud(width=800, height=400, background_color="white").generate(titles)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig)
else:
    st.info("No titles available for the selected range.")

# --- Word Cloud (Abstracts) ---
st.subheader("Word Cloud of Abstracts")
abstracts = " ".join(pd.Series(df_filtered["abstract"]).dropna().astype(str).tolist())
if abstracts.strip():
    wc = WordCloud(width=800, height=400, background_color="white").generate(abstracts)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig)
else:
    st.info("No abstracts available for the selected range.")
