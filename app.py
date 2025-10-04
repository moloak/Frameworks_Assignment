# Building a streamlit application
from data import df 
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS


# Page setup
st.set_page_config(page_title="COVID-19 Research Dashboard", layout="wide")

st.title("COVID-19 Research Dashboard")
st.write("Explore publication trends, sources, and keywords in COVID-19 related research papers.")


# Load data
# Show a preview
st.subheader("Sample of the dataset")
st.dataframe(df.sample(10))  # show first 10 rows

# Interactive filters
years = sorted(df['publish_year'].unique())
min_year, max_year = st.slider("Select publication year range:", 
                               int(min(years)), int(max(years)),
                               (1997, 2012))

source_filter = st.selectbox("Filter by source (optional):", 
                             options=["All"] + list(df['source_x'].unique()[:50]))

# Apply filters
filtered = df[(df['publish_year'].between(min_year, max_year))]
if source_filter != "All":
    filtered = filtered[filtered['source_x'] == source_filter]


# Publications over time
st.subheader("Publications over Time")

papers_per_year = (
    filtered.groupby('publish_year')
    .size()
    .reset_index(name='paper_count')
)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(papers_per_year['publish_year'], papers_per_year['paper_count'], marker='o')
ax.set_title("Number of Publications Over Time")
ax.set_xlabel("Year")
ax.set_ylabel("Number of Publications")
ax.grid(True)
st.pyplot(fig)


# Word Cloud of Titles
import warnings
from collections import Counter
import numpy as np

# Suppress FutureWarnings (like swapaxes deprecation)
warnings.simplefilter(action='ignore', category=FutureWarning)

st.subheader("☁️ Word Cloud of Research Titles")

# Slider to control number of words
max_words = st.slider("Select number of words in WordCloud:", 100, 1000, 500, step=50)

stopwords = set(STOPWORDS)
stopwords.update([
    'study', 'analysis', 'effect', 'using', 'based', 
    'case', 'new', 'approach', 'research'
])

# Break titles into smaller chunks to avoid memory issues
chunks = np.array_split(filtered['title'].astype(str), 10)

word_counts = Counter()
wc_tmp = WordCloud(stopwords=stopwords)

for chunk in chunks:
    text = " ".join(chunk.tolist())
    word_counts.update(wc_tmp.process_text(text))

# Generate word cloud from frequencies (efficient way)
wordcloud = WordCloud(
    width=1200,
    height=600,
    background_color='white',
    stopwords=stopwords,
    colormap='viridis',
    collocations=True,
    max_words=max_words   # now controlled by slider
).generate_from_frequencies(word_counts)

fig, ax = plt.subplots(figsize=(12, 6))
ax.imshow(wordcloud, interpolation='bilinear')
ax.axis('off')
st.pyplot(fig)
