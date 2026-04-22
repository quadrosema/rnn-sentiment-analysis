#class for showing results
import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

MODEL_NAMES = [
    "A",
    "Atrans",
    "B",
    "Btrans"
]

#func that maps and shows the results
def evaluate(data):
    print(f"\n\033[95m[GENERATING FINAL REPORT ASSETS]\033[0m")
    
    xtest = data['xtest']
    ytest = data['ytest']
    
    final = {}

    #setup
    fig , axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for i, name in enumerate(MODEL_NAMES):
        filename = f"{name}.keras"
        
        if os.path.exists(filename):
            print(f"   ⚡ Loading {name}...", end="")
            
            #load
            model = tf.keras.models.load_model(filename)
            
            loss, acc = model.evaluate(xtest, ytest, verbose=0)
            final[name] = acc * 100
            
            #conf matrix
            y_pred_prob = model.predict(xtest, verbose=0)
            y_pred = (y_pred_prob > 0.5).astype("int32")
            
            cm = confusion_matrix(ytest, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i])
            axes[i].set_title(f"{name}\nAcc: {acc:.1%}")
            axes[i].set_ylabel('Actual')
            axes[i].set_xlabel('Predicted')
            
            print(f" Done! (Acc: {acc:.1%})")
        else:
            print(f"   ⚠️ File {filename} not found.")
            final[name] = 0

    plt.tight_layout()
    plt.savefig("Final_Confusion_Matrices.png")
    print("\n📊 Saved 'Final_Confusion_Matrices.png'")

    #bar chart
    plt.figure(figsize=(10, 6))
    names = list(final.keys())
    values = list(final.values())
    
    bars = plt.bar(names, values, color=['#ff9999','#66b3ff','#99ff99','#ffcc99'])
    plt.ylabel('Accuracy (%)')
    plt.ylim(0, 100)
    plt.title('Final Model Comparison')
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f"{yval:.1f}%", ha='center', fontweight='bold')
        
    plt.savefig("Final_Accuracy_Chart.png")
    print("📊 Saved 'Final_Accuracy_Chart.png'")
    
    print(f"\n\033[92m✅ REPORT GENERATION COMPLETE.\033[0m")