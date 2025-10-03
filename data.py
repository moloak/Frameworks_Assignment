import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
import re
from wordcloud import WordCloud, STOPWORDS

# load data into dataframe
df = pd.read_csv('metadata.csv', low_memory=False)

# load first five rows
print(df.head())

# Dataframe dimensions
print(df.shape)

# Data types of columns
print(df.dtypes)

# Generate basic statistics
df.describe()

# Missing values
print(df.isna())

# Data cleaning and preparation
df = df.dropna(axis=1, thresh=len(df) /2) # Drop columns with more than 50% missing values
df = df.fillna({
    'cord_uid' : 'absent',
    'source_x' : 'POO',
    'sha' : 'absent',
    'title' : 'No Title',
    'doi' : '10.1186/rr61',
    'license' : 'no-cc',
    'abstract' : 'Unavailable abstract',
    'publish_time' : '2050-09-24',
    'authors' : 'No author(s)',
    'journal' : 'No journal available',
    'url' : 'https://placeholderwebsite.com'
})
df = df.drop(columns=['pmc_json_files', 'pdf_json_files', 's2_id'], errors='ignore')
cols_to_convert = ['cord_uid', 'sha', 'source_x', 'title', 'doi', 'license', 'abstract', 'publish_time', 'authors', 'journal', 'url']
df[cols_to_convert] = df[cols_to_convert].astype("string")

# Convert date columns to datetime format
df['publish_time'] = pd.to_datetime(df['publish_time'], format = 'mixed')

# Extract year from publication date
df['publish_year'] = df['publish_time'].dt.year
print(df.head())

# Papers by publication year
papers_per_year = df.groupby('publish_year').size().reset_index(name='paper_count')
print(papers_per_year)

# Creating bar chart of top publishing journals
# Count papers published per journal
top_journals = (df['journal'].value_counts().reset_index().rename(columns={'index': 'journal', 'journal': 'paper_count'})
)

# Show top 10 journals
print(top_journals.head(11))


# Find most frequent words
titles = df['title'].dropna().str.lower() # Drop NaN titles and convert to lowercase

# Tokenize: split by words (include even 1-2 letter words)
words = []
for title in titles:
    words.extend(re.findall(r'\b[a-z]+\b', title))  # keep all words

# Count frequency
word_counts = Counter(words)

# Get top 20 most common words
top_words = word_counts.most_common(20)

# Convert to DataFrame for easy view
top_words_df = pd.DataFrame(top_words, columns=['word', 'count'])
print(top_words_df)

# Filter filled data (Exclude 2050)
filtered = df[df['publish_year'] != 2050]

# Count publications per year
papers_per_year = (
    filtered.groupby('publish_year')
    .size()
    .reset_index(name='paper_count')
)

# Plot
plt.figure(figsize=(10,6))
plt.plot(papers_per_year['publish_year'], papers_per_year['paper_count'], marker='o')
plt.title("Number of Publications Over Time (Excluding 2050)")
plt.xlabel("Year")
plt.ylabel("Number of Publications")
plt.grid(True)
plt.savefig('images/plot1.png')

# Word cloud of paper titles
# Collect all titles (drop NaN and placeholders)
titles_text = " ".join(df['title'].dropna().astype(str).tolist())

# Define stopwords (common words to ignore)
stopwords = set(STOPWORDS)

stopwords.update([
    'study', 'analysis', 'effect', 'using', 'based', 
    'case', 'new', 'approach', 'research'
])

# Generate the word cloud
wordcloud = WordCloud(
    width=1200,
    height=600,
    background_color='white',
    stopwords=stopwords,
    colormap='viridis',
    collocations=True  # combine frequent word pairs
).generate(titles_text)

# Plot it
plt.figure(figsize=(15, 8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title("Word Cloud of Research Paper Titles", fontsize=18)
plt.savefig('images/plot2.png')

# Distribution plot of paper count by source
top = df['source_x'].value_counts().nlargest(15)  # Series: index=source, values=counts

plt.figure(figsize=(12,6))
plt.barh(top.index[::-1], top.values[::-1])  # flip so largest is on top
plt.title("Top 15 Sources by Number of Papers")
plt.xlabel("Paper Count")
plt.ylabel("Source")
plt.grid(axis='x', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('images/plot3.png')
