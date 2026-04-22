#this class loads and setups glove
import os
import numpy as np


#config
vocab = 20000 
mlen = 200      
emb = 100

#path setup
base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
glovep= os.path.join(base , "data" , "glove.txt")

#function that for loading and setting up 
def glove(index):
    
    print(f"   - Loading GloVe Embeddings from {glovep}...")

    embeddings = {}
    try:
        with open(glovep, encoding='utf-8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings[word] = coefs
        
        print(f"     Found {len(embeddings)} word vectors.")
    except FileNotFoundError:
        print(f"\033[91m     ❌ ERROR: GloVe file not found at {glovep}\033[0m")
        return None
    
    #creating matrix
    matrix = np.zeros((vocab, emb))
    hits = 0
    misses = 0

    for word, i in index.items():
        if i < vocab:
            vector = embeddings.get(word)
            if vector is not None:
                matrix[i] = vector
                hits += 1
            else:
                misses += 1
                
    print(f"     Matrix Ready: {hits} words found, {misses} missing.")
    return matrix



