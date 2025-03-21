import pandas as pd
from sentence_transformers import SentenceTransformer, util
import joblib

df = pd.read_csv("E:\Assignment\pythonProject\DataNeuron_Text_Similarity.csv")

df.columns = ['text1', 'text2']

model = SentenceTransformer('all-MiniLM-L6-v2')

embeddings1 = model.encode(df['text1'].tolist(), convert_to_tensor=True)
embeddings2 = model.encode(df['text2'].tolist(), convert_to_tensor=True)

similarity_scores = util.cos_sim(embeddings1, embeddings2).diagonal()

df['similarity_score'] = similarity_scores.cpu().numpy()

joblib.dump(model, 'similarity_model.joblib')
df.to_csv("predicted_similarity.csv", index=False)