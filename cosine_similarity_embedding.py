
import os
from google.cloud import firestore
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

file_path = "szakdolgozat-a9498-c9d93c04fb55.json" 
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = file_path
firestore_client = firestore.Client()

collection = firestore_client.collection("events")
docs = collection.stream()

data = []
doc_ids = []

for doc in docs:
    doc_data = doc.to_dict()
    data.append({
        "id": doc.id,
        "description": doc_data.get("description"),
    })
    doc_ids.append(doc.id)

df = pd.DataFrame(data)

tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(df['description'])

embeddings = tfidf_matrix.toarray()

collection = firestore_client.collection("events")

for i in range(len(doc_ids)):
    doc_ref = collection.document(doc_ids[i])

    doc = doc_ref.get()

    if doc.exists and 'embedding_field' not in doc.to_dict():
        doc_ref.update({
            "embedding_field": embeddings[i].tolist()
        })
