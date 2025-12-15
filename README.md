# Multi-Label Text Classification of Research Articles: Evaluating the Performance of Pre-Trained and Transformer Models

## 👥 Contributors
* **Sandeep Pandellapalli** (psp5)
* **Maithili Lingala** (msl10)

## 📌 Project Overview
This project explores **Multi-Label Text Classification** to automate the categorization of academic research articles. Using a dataset of over 20,000 research article abstracts, we performed a comparative analysis between traditional Machine Learning models (Logistic Regression, Naive Bayes, SVM) and advanced Transformer-based models (**RoBERTa**).

The goal was to assign one or more specific labels (e.g., Computer Science, Physics, Mathematics) to an article based solely on its abstract text, improving the accessibility and categorization of academic publications.

## 🎯 Objectives
* **Automate Categorization:** Tag research papers with multiple relevant topics.
* **Compare Techniques:** Evaluate **Feature Extraction** (Bag of Words vs. TF-IDF) and **Multi-Label Transformation** (Binary Relevance, Classifier Chains, Label Powerset).
* **Advanced Modelling:** Assess the performance of **RoBERTa** (Robustly Optimized BERT Approach) against traditional ML baselines.
* **Address Imbalance:** Implement strategies such as Class Weights, Dropout, and Weight Decay to handle label imbalance and overfitting.

## 📂 Dataset Overview
* **Source:** Analytics Vidhya Hackathon
* **Total Entries:** 20,972 Research Articles
* **Features:** `ID`, `Title`, `Abstract`
* **Target Labels:**
    * Computer Science
    * Physics
    * Mathematics
    * Statistics
    * Quantitative Biology
    * Quantitative Finance

*Note: The dataset exhibits significant label imbalance, with Computer Science and Physics being the most represented fields, while Quantitative Finance is underrepresented.*

## 🛠️ Methodology

### 1. Data Preprocessing
To clean and standardize the input text, we applied:
* **Cleaning:** Lowercasing, removing non-alphanumeric characters, and whitespace removal.
* **NLP Techniques:** Stopword removal (NLTK) and Stemming (Snowball Stemmer).
* **Tokenization:** Word piece tokenization was used specifically for the RoBERTa model.

### 2. Approach 1: Traditional Machine Learning
We experimented with different combinations of feature extraction and classification techniques:
* **Feature Extraction:**
    * **Count Vectorizer (Bag of Words):** Focused on word frequency.
    * **TF-IDF:** Focused on word importance relative to the corpus.
* **Multi-Label Strategies:**
    * **Binary Relevance:** Treats each label as an independent binary classification task.
    * **Classifier Chains:** Links classifiers to capture correlations between labels.
    * **Label Powerset:** Transforms unique label combinations into distinct classes.
* **Models:** Logistic Regression, Naive Bayes, Linear SVC.

### 3. Approach 2: Transformer Model (RoBERTa)
We utilized **RoBERTa**, a robustly optimized BERT approach trained on 160GB of text. To counter overfitting and label imbalance, we implemented:
* **Dropout (0.5):** To prevent overfitting.
* **Class Weights:** Adjusted inversely proportional to class frequencies to handle imbalance.
* **Gradient Clipping:** To prevent exploding gradients.
* **Weight Decay:** For regularization.

## 📊 Key Results

### Machine Learning Models
* **Logistic Regression** combined with **TF-IDF** and **Classifier Chains** offered the best balance of metrics (F1 Score ~0.79).
* **Naive Bayes** achieved high recall but suffered from lower precision due to independence assumptions.
* **Impact of Preprocessing:** Preprocessing significantly improved accuracy, recall, and F1 scores across all models.

### Transformer Model (RoBERTa)
* The baseline RoBERTa model initially showed signs of overfitting (high training accuracy vs. lower validation accuracy).
* After applying improvement strategies (Dropout, Weight Decay), the model demonstrated better generalization and stability.
* **MCC (Matthews Correlation Coefficient)** was used as a key metric to ensure the model performed well despite class imbalances.

## 📉 Evaluation Metrics
We utilized metrics specifically suited for multi-label tasks:
* **Hamming Loss:** The fraction of incorrect labels to the total number of labels.
* **Jaccard Score:** Intersection over Union of predicted vs. true labels.
* **MCC Score:** A balanced measure for imbalanced datasets.
* Standard Metrics: Accuracy, Precision, Recall, F1-Score.

## 🚀 Future Scope
To further address label imbalance, we propose a data augmentation strategy using **Large Language Models (LLMs)** like Llama or GPT-4. By systematically rephrasing text data for underrepresented columns (e.g., Quantitative Finance), we can enrich the dataset with linguistically diverse examples without altering the labels, thereby improving model generalization.

## 📚 References
1. N. K. Mishra and P. K. Singh, “Feature construction and SMOTE-based imbalance handling for multi-label learning,” *Journal of Computational Science*, 2021.
2. B. R. Bhamare and J. Prabhu, “A multilabel classifier for text classification and enhanced BERT system,” *ResearchGate*, 2021.
3. J. Nam et al., “Large-scale multi-label text classification — Revisiting neural networks,” 2013.
