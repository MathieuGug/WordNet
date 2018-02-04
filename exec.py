import utils
from sklearn.feature_extraction.text import TfidfVectorizer

import pandas as pd
import pickle

#####################################
#        COMPUTE SIMILARITY         #
#####################################

# If the data are pickled :
SENTENCES1_DUMP_PATH = ''
SENTENCES2_DUMP_PATH = ''
WORDNET_GRAPH_PATH = ''

# Otherwise :
TESTSET_PATH = ''
test = pd.read_csv('', sep=',', header=None)

s1_test = [list(process_sentence(dd)) for dd in test.iloc[:,3]]
s2_test = [list(process_sentence(dd)) for dd in test.iloc[:,4]]


#with open(SENTENCES_1_PATH, 'rb') as fp:
#    s1_test = pickle.load(fp)
#with open(SENTENCES_2_PATH, 'rb') as fp:
#    s2_test = pickle.load(fp)
    
#with open(WORDNET_GRAPH_PATH, 'rb') as fp:
#    G = pickle.load(fp)
#print('Data loaded')


W = nx.adjacency_matrix(G)
M = W/sparse.linalg.norm(W, ord=np.inf)
print('Adjacency matrix computed.')

# Create the tfidf vectorizer for the set of sentences1 and sentences2
vectorizer = TfidfVectorizer(tokenizer = lambda doc : doc.split(" "))
vectorizer.fit([' '.join(sent) for sent in s1_test + s2_test])
tfidf1 = vectorizer.transform(' '.join(sent) for sent in s1_test)
tfidf2 = vectorizer.transform(' '.join(sent) for sent in s2_test)
print('TFIDF calculated.')

# The distributions
D1 = generate_features(s1_test, vectorizer, tfidf1, G, M)
D2 = generate_features(s2_test, vectorizer, tfidf1, G, M)

# Compute the similarity for each distribution
n = len(D1)
sim = np.zeros(n)
for i in range(n):
    sim[i] = dice_similarity(D1[i,], D2[i,])