# Part 1: Data Loading and Basic Exploration
import pandas as pd

# Load dataset (use a sample if too large)
df = pd.read_csv("metadata.csv")

# Inspect first few rows
print(df.head())

# Check dimensions
print("Shape:", df.shape)

# Check column info
print(df.info())

# Check missing values
print(df.isnull().sum())

# Handle missing values: drop rows with no title or publish_time
df_clean = df.dropna(subset=["title", "publish_time"])

# Convert publish_time to datetime
df_clean["publish_time"] = pd.to_datetime(df_clean["publish_time"], errors="coerce")

# Extract year
df_clean["year"] = df_clean["publish_time"].dt.year

# Example new feature: word count of abstract
df_clean["abstract_word_count"] = df_clean["abstract"].fillna("").apply(lambda x: len(x.split()))

print(df_clean[["title", "publish_time", "year", "abstract_word_count"]].head())

import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from wordcloud import WordCloud

# 1. Count papers by year
year_counts = df_clean["year"].value_counts().sort_index()

plt.figure(figsize=(8,5))
plt.bar(year_counts.index, year_counts.values)
plt.title("Publications by Year")
plt.xlabel("Year")
plt.ylabel("Number of Papers")
plt.show()

# 2. Top journals
top_journals = df_clean["journal"].value_counts().head(10)

plt.figure(figsize=(10,6))
sns.barplot(x=top_journals.values, y=top_journals.index, palette="viridis")
plt.title("Top 10 Journals")
plt.xlabel("Number of Papers")
plt.show()

# 3. Word cloud of titles
titles = " ".join(df_clean["title"].dropna().tolist())
wordcloud = WordCloud(width=800, height=400, background_color="white").generate(titles)

plt.figure(figsize=(10,6))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("Word Cloud of Paper Titles")
plt.show()

# 4. Distribution by source
source_counts = df_clean["source_x"].value_counts().head(10)

plt.figure(figsize=(10,6))
sns.barplot(x=source_counts.values, y=source_counts.index, palette="coolwarm")
plt.title("Top Sources of Papers")
plt.xlabel("Count")
plt.show()

