#class that actually runs the models
import models
import os
import train
import matplotlib.pyplot as plt

#the func that does run
def run(data):
 
    xtrain = data['xtrain']
    ytrain = data['ytrain']
    xval = data['xval']
    yval = data['yval']
    matrix = data['matrix']

    results = {}

    #model A without transfer
    print("\n--- Exp 1/4: model A ---")
    modela = models.build('A', transfer=False)
    hista = train.train_model(modela, xtrain, ytrain, xval, yval, "A")
    results['A_Scratch'] = hista.history['val_accuracy'][-1]
    modela.save("A.keras")

    #model A with transfer
    print("\n--- Exp 2/4: model A treansfer ---")
    transa = models.build('A', transfer=True, matrix=matrix)
    histtransa = train.train_model(transa, xtrain, ytrain, xval, yval, "Atrans")
    results['A_Transfer'] = histtransa.history['val_accuracy'][-1]
    transa.save("Atrans.keras")

    #model B without transfer
    print("\n--- Exp 3/4: model B ---")
    modelb = models.build('B', transfer=False)
    histb = train.train_model(modelb, xtrain, ytrain, xval, yval, "B")
    results['B_Scratch'] = histb.history['val_accuracy'][-1]
    modelb.save("B.keras")

    #model B with transfer
    print("\n--- Exp 4/4: model B transfer ---")
    transb = models.build('B', transfer=True, matrix=matrix)
    histtransb = train.train_model(transb, xtrain, ytrain, xval, yval, "Btrans")
    results['Btrans'] = histtransb.history['val_accuracy'][-1]
    transb.save("Btrans.keras")

    print("\nall experiments finished!")
    print(results)