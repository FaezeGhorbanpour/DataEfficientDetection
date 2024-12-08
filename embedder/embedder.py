from sentence_transformers import SentenceTransformer
from FlagEmbedding import BGEM3FlagModel

class Embedder:
    def __init__(self, name='LaBSE', d=None):
        self.name = name
        self.d = d

        if self.name == 'labse':
            self.model = SentenceTransformer('sentence-transformers/LaBSE')
        elif self.name.lower() == 'minilm':
            self.model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        elif self.name == 'm3':
            self.model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)
        elif self.name.lower() == 'e5':
            self.model = SentenceTransformer('intfloat/multilingual-e5-large')





    def embed(self, sentences):
        embeddings = self.model.encode(sentences)
        return embeddings


    def similarity(self, embeddings1, embeddings2):
        similarities = self.model.similarity(embeddings1, embeddings2)
        return similarities
