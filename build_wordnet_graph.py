import utils
import wordnet as wn
import pickle
from sklearn.feature_extraction.text import CountVectorizer

###########################################
#            WORDNET GRAPH NODES          #
###########################################
G = nx.DiGraph()

# 1. We go through every synsets in wordnet and add them as nodes
# with their gloss as attributes and types (synset / tokenPOS / token)
# 1 hour on my machine
for synset in list(wn.all_synsets()):
    #synset, tokenPOS and token nodes
    syn = synset.name()
    
    # To avoid the .22_caliber.a.01, .38_caliber.a.01 and .45_caliber.a.01
    if syn[0] == '.' or len(synset.name().split('.')) > 3:
        continue
        
    tokenPOS = '.'.join(synset.name().split('.')[0:2])
    token = synset.name().split('.')[0]
    
    G.add_node(syn, type="synset", gloss=wn.synset(syn).definition())
    
    if tokenPOS not in G.nodes():
        G.add_node(tokenPOS, type="tokenPOS")
    
    if token not in G.nodes():
        G.add_node(token, type="token")
        

#######################################
#      WORDNET EDGES FROM GLOSSES     #
#######################################

#2. We build a corpus of all the glosses by tagging every word
gloss_dict = {}
i = 0
for node, attr in G.nodes(data=True):
    if ('gloss' in attr.keys()):
        #if (i % 1000 == 0): print(str(i) + '/' + str(len(G.nodes())) )
        i = i + 1
        gloss_dict[node] = ' '.join(list(tag_sentence(list(process_sentence(attr['gloss'])))))

# Since w(ij) = exp(-(log(ri) - mu)/2sigma**2, we make a CountVectorizer over the tagged glosses
gloss_vectorizer = CountVectorizer(tokenizer=lambda doc: doc.split(' '))
gloss_vectorizer.fit(gloss_dict.values())
counter = gloss_vectorizer.transform(gloss_dict.values())

# The word count of every words
freq = np.sum(counter, axis=1)

# We compute the edge weight between the synset -> tokenPOS of the gloss
mu = np.log(np.mean(freq))
sigma = np.log(np.std(freq))

for synset in gloss_dict.keys():
    for tokenPOS in gloss_dict[synset].split(' '):
        word_ind = gloss_vectorizer.vocabulary_[tokenPOS]
        log_freq = np.log(freq[word_ind])
        w = np.exp(-(log_freq - mu)**2/(2*sigma**2))
        G.add_edge(synset, tokenPOS, weight = float(w))
        
#######################################
#       COMPLETE WORDNET GRAPH        #
#######################################

# About 6 hours on my machine
for node, attr in G.nodes(data=True):
    #print(node)
    # Indication of time
    #if (i%100 == 0):
    #    print(str(i) + ' out of ' + str(G.number_of_nodes()))
    #i += 1
    
    # Semantical relationships
    if attr['type'] == 'synset':
        synset = wn.synset(node)
        for hypernym in synset.hypernyms():
            G.add_edge(node, hypernym.name(), weight=.5)
        for hyponym in synset.hyponyms():
            G.add_edge(node, hyponym.name(), weight=.5)
        for member_meronym in synset.member_meronyms():
            G.add_edge(node, member_meronym.name(), weight=.5)
        for member_holonym in synset.member_holonyms():
            G.add_edge(node, member_holonym.name(), weight=.5)
            
        # On lemmas
        for lemma in synset.lemmas():
            for antonym in lemma.antonyms():
                G.add_edge(node, antonym.synset().name(), weight=.5)
            for pertainyms in lemma.pertainyms():
                G.add_edge(node, pertainyms.synset().name(), weight=.5)
                
    if attr['type'] == 'tokenPOS':
        # tokenPOS -> synsets
        token, pos = node.split('.')
        synsets = wn.synsets(token, pos=pos)
        synsets_names = filter(lambda x: x in G.nodes(),
                               [s.name() for s in synsets])
        
        sum_weight = 0
        for synset in synsets_names:
            #print(synset)
            weight = sum([l.count() for l in wn.synset(synset).lemmas()]) + .1
            G.add_edge(node, synset,
                        weight = weight)
            sum_weight += weight
                
        # token -> tokenPOS
        G.add_edge(token, node, weight = sum_weight)
        
        # synset -> synset with common tokenPOS
        # Generate permutations for all the synsets with the shared TokenPOS
        synsets_edges = permutations(synsets_names, 2)
        
        for syn_edge in synsets_edges:
            #print(syn_edge)
            # Create the edge
            G.add_edge(syn_edge[0], syn_edge[1])
            # If it already exists, update it
            if 'weight' in G.edges[syn_edge].keys():
                weight = G.edges[syn_edge]['weight']
                G.add_edge(syn_edge[0], syn_edge[1], weight = weight + 1)
            else:
                G.add_edge(syn_edge[0], syn_edge[1], weight = 1)
                
                
# Save the graph for posterior use
with open('wordnet.graph', 'wb') as fp:
     pickle.dump(G, fp)