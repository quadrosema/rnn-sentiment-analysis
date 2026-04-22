# RNN Sentiment Analysis (LSTM vs Bi-GRU + GloVe)

This project builds and evaluates recurrent neural network models for binary sentiment analysis on movie reviews.

## Problem
Binary text classification:
- Positive
- Negative

## Models
- LSTM (Model A)
- Bidirectional GRU (Model B)

Each model was tested in two settings:
- training from scratch
- transfer learning using pretrained GloVe embeddings

## Data Processing
- CSV loading with pandas
- text tokenization using Keras Tokenizer
- sequence padding to fixed length
- label encoding
- train/validation/test split (80/10/10)

## Transfer Learning
Pretrained GloVe embeddings were loaded into the embedding layer to compare learned embeddings vs fixed pretrained word vectors.

## Results

### Accuracy Comparison
![Accuracy Comparison](results/Final_Accuracy_Chart.png)

### Confusion Matrices
![Confusion Matrices](results/Final_Confusion_Matrices.png)

### Final Accuracy
- LSTM (Scratch): 89.5%
- LSTM (GloVe): 87.8%
- Bi-GRU (Scratch): 89.6%
- Bi-GRU (GloVe): 90.0%

## Key Insights
- Bi-GRU with GloVe achieved the best overall accuracy
- pretrained embeddings can improve semantic representation
- model architecture had a measurable effect on classification quality

## Project Structure
```text
rnn-sentiment-analysis/
├─ src/
│  ├─ dloader.py
│  ├─ experiments.py
│  ├─ gloader.py
│  ├─ main.py
│  ├─ models.py
│  ├─ results.py
│  ├─ train.py
├─ data/
├─ results/
│  ├─ Final_Accuracy_Chart.png
│  ├─ Final_Confusion_Matrices.png
├─ README.md
├─ requirements.txt
├─ .gitignore