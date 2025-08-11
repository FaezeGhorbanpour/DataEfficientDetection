from bertopic import BERTopic
from deep_translator import GoogleTranslator
from scipy.stats import entropy
import scipy
import numpy as np
import scipy.linalg

# Monkey patch triu into scipy.linalg
scipy.linalg.triu = np.triu

from gensim.models import CoherenceModel
from gensim.corpora import Dictionary
import numpy as np
import pandas as pd


class TopicModeler:
    def __init__(self, embed_model, language='auto', target_lang='en'):
        self.embed_model = embed_model
        self.language = language
        self.target_lang = target_lang
        self.topic_model = BERTopic(embedding_model=self.embed_model)
        self.topics = None
        self.probs = None
        self.topic_info = None
        self.labels = None
        self.texts = None
        self.tokenized_texts = None

    def __call__(self, dataset_texts, labels=None, tokenized_texts=None):
        self.texts = dataset_texts
        self.tokenized_texts = tokenized_texts
        self.topics, self.probs = self.topic_model.fit_transform(dataset_texts)
        self.topic_info = self.topic_model.get_topic_info()
        self.labels = np.array(labels) if labels is not None else None
        return self.topics

    def count_dominant_topics(self, min_ratio=0.01):
        total_docs = sum(self.topic_info["Count"])
        dominant = self.topic_info[
            (self.topic_info["Topic"] != -1) & (self.topic_info["Count"] >= min_ratio * total_docs)
        ]
        return len(dominant), dominant

    def compute_entropy(self):
        topic_entropy = entropy(np.array(self.probs).T)
        return np.mean(topic_entropy)

    def compute_entropy_per_label(self):
        results = {}
        for lbl in [0, 1]:
            idx = self.labels == lbl
            ent = entropy(np.array(self.probs)[idx].T)
            results[lbl] = np.mean(ent)
        return results

    def compute_purity(self):
        top_topic_probs = np.max(self.probs, axis=1)
        return np.mean(top_topic_probs)

    def compute_purity_per_label(self):
        results = {}
        for lbl in [0, 1]:
            idx = self.labels == lbl
            top_probs = np.max(self.probs[idx], axis=1)
            results[lbl] = np.mean(top_probs)
        return results

    def translate_topics(self, top_n=10):
        translator = GoogleTranslator(source=self.language, target=self.target_lang)
        translated = {}
        for topic_num in self.topic_info["Topic"]:
            if topic_num == -1:
                continue
            topic_words = self.topic_model.get_topic(topic_num)[:top_n]
            translated[topic_num] = [translator.translate(w) for w, _ in topic_words]
        return translated

    def get_topic_sizes(self):
        return self.topic_info[["Topic", "Count"]]

    def get_topic_keywords(self, top_n=10):
        keywords = {}
        for topic_num in self.topic_info["Topic"]:
            if topic_num == -1:
                continue
            topic_words = self.topic_model.get_topic(topic_num)[:top_n]
            keywords[topic_num] = [w for w, _ in topic_words]
        return keywords

    def get_topic_sizes_per_label(self):
        df = pd.DataFrame({'topic': self.topics, 'label': self.labels})
        sizes = df.groupby(['label', 'topic']).size().reset_index(name='count')
        return sizes[sizes.topic != -1]

    def get_topic_keywords_per_label(self, top_n=10):
        df = pd.DataFrame({'topic': self.topics, 'label': self.labels})
        result = {}
        for lbl in [0, 1]:
            topics_lbl = df[df.label == lbl].topic.value_counts().index.tolist()
            result[lbl] = {}
            for t in topics_lbl:
                if t == -1:
                    continue
                words = self.topic_model.get_topic(t)[:top_n]
                result[lbl][t] = [w for w, _ in words]
        return result

    def compute_gensim_coherence(self, top_n=10):
        if self.tokenized_texts is None:
            raise ValueError("Tokenized texts are required for coherence calculation.")
        topics = self.get_topic_keywords(top_n=top_n)
        topic_words = list(topics.values())
        dictionary = Dictionary(self.tokenized_texts)
        corpus = [dictionary.doc2bow(text) for text in self.tokenized_texts]
        cm = CoherenceModel(topics=topic_words, texts=self.tokenized_texts,
                            dictionary=dictionary, coherence='c_v')
        return cm.get_coherence()

    def auto_reduce_topics(self, min_topics=5, max_topics=30, step=5, top_n=10):
        if self.texts is None or self.tokenized_texts is None:
            raise ValueError("Model must be called first with tokenized texts.")
        best_score = -1
        best_model = None
        best_n = None

        for n in range(min_topics, max_topics + 1, step):
            reduced_model = self.topic_model.reduce_topics(self.texts, nr_topics=n)
            temp_model = TopicModeler(self.embed_model)
            temp_model.topic_model = reduced_model
            temp_model.texts = self.texts
            temp_model.tokenized_texts = self.tokenized_texts
            temp_model.topics, temp_model.probs = reduced_model.transform(self.texts)
            temp_model.topic_info = reduced_model.get_topic_info()
            score = temp_model.compute_gensim_coherence(top_n=top_n)
            if score > best_score:
                best_score = score
                best_model = reduced_model
                best_n = n

        self.topic_model = best_model
        self.topics, self.probs = self.topic_model.transform(self.texts)
        self.topic_info = self.topic_model.get_topic_info()
        return best_n, best_score
