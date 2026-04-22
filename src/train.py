#class that trains the the modles
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

#func that trains the model and allows it to learn
def train_model(model, xtrain, ytrain, xval, yval, name):
    
    print(f"\n[TRAINING START] {name}...")
    
    #callback
    stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=0.00001)


    history = model.fit(
        xtrain, ytrain,
        validation_data=(xval, yval),
        epochs=10,           
        batch_size=128,       
        callbacks=[stop, lr],
        verbose=1             
    )
    
    print(f"   [TRAINING COMPLETE] {name}")
    return history