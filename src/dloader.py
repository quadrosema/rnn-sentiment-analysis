#this class loads and cleans the data
import os
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
import gloader

#config
vocab = 20000 
mlen = 200      
emb = 100

#path setup
base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
csv = os.path.join(base , "data" , "dataset.csv")

#func that reads the data
def read():
    
    print(f"   - Reading CSV from {csv}...")    
    df = pd.read_csv(csv)

    #rename columns to standard text and label
    if 'review' in df.columns:
        df.rename(columns={'review': 'text', 'sentiment': 'label'}, inplace=True)

    #splitting
    train_text, temp_text, train_labels, temp_labels = train_test_split(
        df['text'] , 
        df['label'], 
        train_size=0.8, 
        random_state=42
    )

    val_text, test_text, val_labels, test_labels = train_test_split(
        temp_text, 
        temp_labels, 
        train_size=0.5,
        random_state=42
    )

    #tokenization
    print("   - Fitting Tokenizer...")
    tokenizer = Tokenizer(num_words=vocab, oov_token="<OOV>")
    tokenizer.fit_on_texts(train_text)

    xtrain = pad_sequences(tokenizer.texts_to_sequences(train_text), maxlen=mlen)
    xval = pad_sequences(tokenizer.texts_to_sequences(val_text), maxlen=mlen)
    xtest = pad_sequences(tokenizer.texts_to_sequences(test_text), maxlen=mlen)


    #encoding
    le = LabelEncoder()
    ytrain = le.fit_transform(train_labels)
    yval = le.transform(val_labels)
    ytest = le.transform(test_labels)


    #loading into glove
    print(f"   - Data Shapes: Train {xtrain.shape}, Val {xval.shape}, Test {xtest.shape}")
    matrix = gloader.glove(tokenizer.word_index)

    return {
        'xtrain': xtrain, 'ytrain': ytrain,
        'xval': xval, 'yval': yval,
        'xtest': xtest, 'ytest': ytest,
        'matrix': matrix,
        'size': vocab,
        'mlen': mlen,
        'emb': emb
    }
