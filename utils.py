# download wordnet + wordnet_ic + universal_tagset
#nltk.download()

import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from nltk import word_tokenize, map_tag, pos_tag
import string
import re
import time

from math import log
from itertools import permutations

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse
from scipy.spatial.distance import dice

#####################################
#         TEXT PROCESSING           #
#####################################

def process_sentence(sentence):
    """sentence: list of words split by space"""
    #english_stopwords = set(
    #    [stopword for stopword in stopwords.words('english')])
    punctuation = set(string.punctuation)
    punctuation.update(["``", "`", "..."])
    # WordNet lemmatizer
    Lem = WordNetLemmatizer()

    tokens = list((filter(lambda x: #x.lower() not in english_stopwords and
                        x.lower() not in punctuation,
                        [t.lower() for t in word_tokenize(sentence)
                         if t.isalpha()])))
    
    for word in tokens:
        lemword = Lem.lemmatize(word)
        yield lemword

def tag_to_wordnet(tag):
    """Convert a universal tag to a wordnet tag"""
    if (tag == 'ADJ'): return('a')
    elif (tag == 'ADV'): return('r')
    elif (tag == 'NOUN'): return('n')
    elif (tag == 'VERB'): return('v')
    else: return None

def tag_sentence(proc_sent):
    """proc_sent : sentence returned by process_sentence()"""
    """return a list of the form [('word1.n', 'adj1.a', ...)]
    Useful for the tagging of glosses in wordnet"""
    posTagged = pos_tag(proc_sent)
    for word, tag in posTagged:
        # wordnet style
        wn_tag = tag_to_wordnet(map_tag('en-ptb', 'universal', tag))
        wn_word = wn.morphy(word, wn_tag)
        if wn_word != None and wn_tag != None:
            yield '.'.join([wn_word, wn_tag])
            
#####################################
#     STATIONARY DISTRIBUTIONS      #
#####################################   

def dice_similarity(x, y):
    """The dice similarity"""
    s1 = np.linalg.norm(x, ord=1)
    s2 = np.linalg.norm(y, ord=1)
    return 2*np.sum([min(xi, yi) for xi, yi in zip(x, y)])/(s1 + s2)

def stationary_distribution(sentence, vectorizer, i, tfidf, G, M):
    """return a vector of the same length as the number of nodes
    Input :
    - sentence : list of words
    - vectorizer : the TfidfVectorizer on the entire corpus
    - i : index of the sentence in the corpus
    - tfidf : the tfidf matrix
    - G : the wordnet graph
    - M : the normalized adjacency matrix of the graph
    Output :
    - a (n_nodes) vector representing the stationary distribution for the sentence"""
    voc = vectorizer.vocabulary_
    n = len(G.nodes())
    v_0 = np.zeros(n)
    v = v_0
    v_temp = sparse.csr_matrix(np.zeros(n))
    
    # 1. The initial distribution on each term in the sentence is weighted by its tfidf,
    # and then normalized
    for word in sentence:
        try:
            ind = G.nodes().index(word)
        except ValueError:
            continue
        v_0[ind] = tfidf[i, voc[word]]
    
    v_0_norm = np.linalg.norm(v_0, ord=2)
    if (v_0_norm > 0):
        v_0 = v_0/np.linalg.norm(v_0, ord=2)

    # 2. Stopping criterion
    beta = .1 # probability to go back to the initial distribution
    eps = 1e1
    while (eps > 10e-10):
        #print(v_0.shape, L_rw.shape, v.shape)
        v_temp = beta*v_0 + (1 - beta)*M.dot(v)
        eps = np.linalg.norm(v_temp - v)
        v = v_temp
    return v

def generate_features(sentences, vectorizer, tfidf, G, M):
    """from a given list of sentences (each being a list of words),
    generate the corresponding stationary distributions"""
    n_sentences = len(sentences)
    n_nodes = len(G.nodes())
    
    # The resulting stationary distributions
    D = np.zeros(shape=(n_sentences, n_nodes))
    
    # 2. The stationary distribution for all the sentences
    for i, sent in enumerate(sentences):
        if (i % 100 == 0):
            print(i)
        # Sometime the resulting sentence is empty...
        if len(sent) == 0:
            D[i, :] = np.ones(n_nodes)
        else:
            distrib = stationary_distribution(sent, vectorizer, i, tfidf, G, M)
            D[i, :] = distrib
    return D

