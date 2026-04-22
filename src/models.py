#class for building the models
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, GRU, Dense, Dropout, Bidirectional, SpatialDropout1D

#config
vocab = 20000
mlen = 200
emb = 100

#function that vuilds
def build(type, transfer=False, matrix=None):

    model = Sequential()
    
    #emb layer
    if transfer and matrix is not None:
        print(f"   [BUILDING] Model {type} with GloVe (Transfer Learning)...")

        model.add(Embedding(
            input_dim = vocab, 
            output_dim = emb, 
            weights = [matrix], 
            input_length = mlen, 
            trainable=False)
            ) 
        
    else:
        print(f"   [BUILDING] Model {type} from Scratch...")

        model.add(Embedding(
            input_dim=vocab, 
            output_dim=emb, 
            input_length=mlen)
            )

    #arch
    if type == 'A':

        #lstm
        model.add(SpatialDropout1D(0.2)) 
        model.add(LSTM(64, dropout=0.2)) 
        

    elif type == 'B':

        #gru
        model.add(SpatialDropout1D(0.2))
        model.add(Bidirectional(GRU(64, return_sequences=True))) 
        model.add(Bidirectional(GRU(32)))                        
    
    #classifier head + output
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid')) 
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
