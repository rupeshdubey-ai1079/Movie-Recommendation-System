import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---- Load Dataset ----
# Place 'movies.csv' here
df = pd.read_csv("tmdb_5000_movies.csv")
df = df[["title","overview","genres"]].dropna()

# ---- Content-Based: TF-IDF on Overview ----
tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
tfidf_matrix = tfidf.fit_transform(df["overview"])
cosine_sim   = cosine_similarity(tfidf_matrix, tfidf_matrix)

indices = pd.Series(df.index, index=df["title"]).drop_duplicates()

def content_recommend(title, n=10):
    if title not in indices:
        return f"Movie '{title}' not found."
    idx = indices[title]
    sim_scores = sorted(enumerate(cosine_sim[idx]),
                        key=lambda x: x[1], reverse=True)[1:n+1]
    movie_indices = [i[0] for i in sim_scores]
    return df["title"].iloc[movie_indices].tolist()

# ---- Test ----
print("=== Recommendations for 'The Dark Knight' ===")
recs = content_recommend("The Dark Knight")
for i, r in enumerate(recs, 1):
    print(f"{i}. {r}")

print("\n=== Recommendations for 'Avengers' ===")
recs2 = content_recommend("The Avengers")
for i, r in enumerate(recs2, 1):
    print(f"{i}. {r}")