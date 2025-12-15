# Multi-Label Text Classification of Research Articles

### IS 517: Methods of Data Science (University of Illinois)

## 📌 Project Overview
This project explores **Multi-Label Text Classification** to automate the categorization of academic research articles. Using a dataset of over 20,000 research article abstracts, we performed a comparative analysis between traditional Machine Learning models (Logistic Regression, Naive Bayes, SVM) and advanced Transformer-based models (**RoBERTa**).

The goal was to assign one or more specific labels (e.g., Computer Science, Physics, Mathematics) to an article based solely on its abstract text.

## 👥 Contributors
* **Sandeep Pandellapalli** (psp5)
* **Maithili Lingala** (msl10)

## 🎯 Objectives
* To automate the tagging of research papers to improve discoverability and categorization.
* To compare the effectiveness of **Feature Extraction** methods (Bag of Words vs. TF-IDF).
* To evaluate **Multi-Label Transformation** techniques (Binary Relevance, Classifier Chains, Label Powerset).
* To assess the performance of **Transformer models (RoBERTa)** against traditional ML baselines.

## 📂 Dataset
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

*Note: The dataset exhibits label imbalance, with Computer Science and Physics being the most represented fields.*

## 🛠️ Methodology

### 1. Data Preprocessing
To clean and standardize the input text, we applied the following steps:
* **Cleaning:** Lowercasing, removing non-alphanumeric characters, and whitespace removal.
* **NLP Techniques:** Stopword removal and Stemming/Lemmatization using NLTK and spaCy.
* **Tokenization:** Word piece tokenization for the RoBERTa model.

### 2. Feature Extraction (Traditional ML)
* **Count Vectorizer (Bag of Words):** focused on word frequency.
* **TF-IDF:** focused on word importance relative to the corpus.

### 3. Classification Techniques
We employed three strategies to handle the multi-label nature of the data:
* **Binary Relevance:** Treats each label as a separate binary classification problem.
* **Classifier Chains:** Accounts for label correlations by chaining classifiers.
* **Label Powerset:** Transforms the problem into a multi-class problem by treating unique label combinations as distinct classes.

### 4. Models Implemented
* **Machine Learning:** Logistic Regression, Naive Bayes, Linear SVC.
* **Deep Learning:** RoBERTa (Robustly Optimized BERT Approach).
    * *Optimization strategies used:* Dropout, Class Weights (to handle imbalance), Gradient Clipping, and Weight Decay.

## 📊 Key Results

### Traditional ML Models
* **Logistic Regression** consistently performed well, particularly when combined with **Classifier Chains** and **TF-IDF**.
* **Naive Bayes** showed high recall but struggled with precision.
* **SVM** showed strong precision but had inconsistent recall across different setups.

### Transformer Model (RoBERTa)
* Initial training showed signs of overfitting.
* After applying improvement strategies (Dropout, Weight Decay), the model demonstrated better generalization.
* **MCC (Matthews Correlation Coefficient)** was used as a robust metric for evaluation, ensuring the model performed well despite class imbalances.

## 📉 Evaluation Metrics
We used specific metrics tailored for multi-label classification:
* **Hamming Loss:** The fraction of incorrect labels to the total number of labels.
* **Jaccard Score:** Measures intersection over union for predicted vs. true labels.
* **Standard Metrics:** Accuracy, Precision, Recall, F1-Score.

## 🚀 Future Scope
* **Data Augmentation:** Leveraging LLMs (like Llama or GPT-4) to rephrase text in underrepresented categories to balance the dataset.
* **Architecture Tuning:** Experimenting with different transformer architectures and hyperparameter tuning to further reduce validation loss variability.

## 📚 References
1.  Mishra, N. K., & Singh, P. K. (2021). "Feature construction and SMOTE-based imbalance handling for multi-label learning."
2.  Bhamare, B. R., & Prabhu, J. (2021). "A multilabel classifier for text classification and enhanced BERT system."
3.  Dataset provided by Analytics Vidhya / Kaggle.
