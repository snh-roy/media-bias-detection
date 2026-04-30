## This repository contains the intermediate milestone for a Natural Language Procession course, it is designed to detect subjective bias in news media using Machine Learning models. 

## As of now, this project implements a **Baseline Logistic Regression Model** to classify text as either *Biased* or *Neutral*.

### Key Features:
* **Custom Text Pipeline:** Integrated cleaning using `contractions.py` for text normalization.
* **Feature Engineering:** Utilizes `CountVectorizer` for Unigram and Bigram extraction (10,000 features).
* **Performance:** Achieved **85.7% Accuracy** with balanced F1-scores across both classes.
* **Model Interpretation:** Direct analysis of feature weights ($w_i$) to identify linguistic triggers of bias.


## Future Work 
To advance beyond the baseline and solve the limitations found in word-counting models, the following will be added for the final deliverable:
* **Dense Word Embeddings:** Transitioning from word counts to semantic vector spaces (Word2Vec).
* **Neural Network Architecture:** Implementing a Deep Learning model (LSTM or Transformer) to capture non-linear relationships and long-distance context.

## How to Run Locally
There is **no need to download a CSV file** separately. The project uses the Hugging Face `datasets` library to pull the data directly into memory.

1. **Clone the repository:**
     git clone https://github.com/snh-roy/media-bias-detection


2. **Install dependencies:**
    pip install pandas scikit-learn datasets

3. **Exceute**
Open notebooks/01_baseline_model.ipynb in your IDE and run all cells. 
