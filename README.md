# California Stays review 

## Project Objective

This project develops an end-to-end NLP pipeline to classify hotel reviews as **positive** or **negative**, and to explain model predictions using Explainable AI techniques.

Focus areas:
- Text preprocessing and cleaning
- TF-IDF vs BERT embeddings comparison
- Supervised classification
- Model evaluation and threshold optimization
- Explainability with LIME

Dataset: TripAdvisor hotel reviews (California cities)


## Dataset Overview

- Reviews filtered by selected California cities
- Sentiment labeling:
  - Positive → rating ≥ 4
  - Negative → rating ≤ 2
- Class imbalance handled via stratified sampling

Pre-balancing:
- Positive: 28,819
- Negative: 5,508

Post-balancing:
- 5,508 positive
- 5,508 negative

Train/Test split:
- 90% training
- 10% test


##  Data Cleaning & Processing

Key preprocessing steps:

- Price field cleaning (removal of text expressions and ranges)
- Ratings conversion to numeric
- HTML tag removal from address field
- City extraction and normalization
- Stop-word removal (TF-IDF pipeline)
- Tokenization (TF-IDF / BERT tokenizer)


##  Feature Engineering

Two text representation approaches:

### TF-IDF
- Sparse representation
- Stop-words removed
- Words treated independently

### BERT Embeddings
- Contextual embeddings
- Dense representation
- Semantic awareness
- No stop-word removal


##  Models Evaluated

### Classical ML (TF-IDF features)

- Random Forest
- Support Vector Machine (SVM)
- Feed Forward Neural Network (FFNN)

### Deep Representation (BERT embeddings)

- FFNN classifier on contextual embeddings


##  Model Performance

### TF-IDF

| Model | Accuracy |
|--------|----------|
| Random Forest | 89.8% |
| SVM | 92.8% |
| FFNN | 94.2% |

### BERT + FFNN

- Accuracy: 94.2%

Best overall performance:
- FFNN (both TF-IDF and BERT)
- SVM showed strong generalization in classical ML



##  Explainable AI (XAI)

Explainability implemented using:

- **LIME (Local Interpretable Model-Agnostic Explanations)**

Key insights:

- Positive indicators: *great*, *perfect*, *clean*
- Strong negative signal: *dump*
- BERT explanations less intuitive due to semantic embedding structure
- LIME provides local, instance-level interpretability


##  Tech Stack

- Python
- Scikit-learn
- TensorFlow / Keras
- HuggingFace Transformers
- LIME
- Pandas
- NumPy
- Matplotlib / Seaborn


##  Key Takeaways

- Neural models outperform tree-based methods in text classification.
- Contextual embeddings (BERT) provide richer representations but similar accuracy to optimized TF-IDF pipelines.
- Explainable AI tools like LIME are essential for model transparency in NLP applications.

## Author
Daniele Lepre

Alice Anna Maria Brunazzi

Ernesto Pedrazzini
