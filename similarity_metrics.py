import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import nltk
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util
from smatchpp import Smatchpp


nltk.download("punkt")
import numpy as np
stemmer = PorterStemmer()
model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
smatch_measure = Smatchpp()



# Load spaCy Portuguese model (run: python -m spacy download pt_core_news_md)
nlp = spacy.load("pt_core_news_md")

def lemmatize(text):
    doc = nlp(text.lower())
    lemmas = [token.lemma_ for token in doc if token.is_alpha]
    return " ".join(lemmas)

def jaccard_similarity(a, b):
    set_a, set_b = set(a.split()), set(b.split())
    return len(set_a & set_b) / len(set_a | set_b)

def lemma_based_similarity(text1, text2):
    lemm1 = lemmatize(text1)
    lemm2 = lemmatize(text2)

    return jaccard_similarity(lemm1, lemm2)

def stem_text(text):
    tokens = nltk.word_tokenize(text.lower())
    stems = [stemmer.stem(tok) for tok in tokens if tok.isalpha()]
    return " ".join(stems)

def stem_based_similarity(text1, text2):
    s1 = stem_text(text1)
    s2 = stem_text(text2)

    return jaccard_similarity(s1, s2)


def semantic_similarity(text1: str, text2: str) -> float:
    """
    Computes semantic similarity between two sentences using a multilingual model.
    Returns a cosine similarity score in the range [-1, 1].
    """
    # Encode both texts
    emb1 = model.encode(text1, convert_to_tensor=True)
    emb2 = model.encode(text2, convert_to_tensor=True)

    # Cosine similarity
    score = util.cos_sim(emb1, emb2).item()
    return score

def smatch_similarity(predicted: str, gold: str):
    return smatch_measure.score_pair(predicted,
                                     gold)["main"]

def return_top_k(reference, predictions, field, top_k, criteria): #field might be "amr" or "sent"
    
    list_scores = []
    for instance in predictions:

        predicted = instance[field].strip()
        gold = reference[field].strip()

        if criteria == "lemma":
            score = lemma_based_similarity(predicted, gold)
        elif criteria == "stem":
            score = stem_based_similarity(predicted, gold)
        elif criteria == "cosine":
            score = semantic_similarity(predicted, gold)
        elif criteria == "smatch-recall":
            score = smatch_similarity(predicted, gold)["Recall"]
        elif criteria == "smatch-f1":
            score = smatch_similarity(predicted, gold)["F1"]
        else:
            print(f"{criteria} not supported. Alternatives: lemma, stem, cosine, smatch-recall, smatch-f1")
            break

        list_scores.append(score)

    # Get indices of the top-k largest values, largest first
    top_k_indices = np.argsort(list_scores)[-top_k:][::-1]
    return [(predictions[k], list_scores[k]) for k in top_k_indices]