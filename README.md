# NLP_Text_Classification

# **Text Classification Using Machine Learning**  

## **Project Overview**  
This project focuses on **text classification** using **Logistic Regression** and **Naïve Bayes**. The dataset is preprocessed, transformed using **TF-IDF vectorization**, and evaluated using multiple performance metrics.  

## **Table of Contents**  
- [Project Overview](#project-overview)  
- [Dataset](#dataset)  
- [Preprocessing](#preprocessing)  
- [Model Training](#model-training)  
- [Evaluation](#evaluation)  
- [Results](#results)  
- [Installation](#installation)  
- [Usage](#usage)  
- [Future Improvements](#future-improvements)  
- [License](#license)  

---

## **Dataset**  
The dataset contains textual data with corresponding labels. It is divided into:  
- **training.csv** (80% of data)  
- **test.csv** (20% of data)  

Each record consists of:  
- `text`: The input text  
- `label`: The category/class assigned to the text  

---

## **Preprocessing**  
Before training, the text data undergoes preprocessing:  
✔ Convert text to lowercase  
✔ Tokenization using **NLTK**  
✔ Stopword removal using **NLTK**  
✔ Stemming using **PorterStemmer**  
✔ Lemmatization using **WordNetLemmatizer**  
✔ Feature extraction using **TF-IDF Vectorization**  

---

## **Model Training**  
Two models were trained and evaluated:  

1️⃣ **Logistic Regression**  
2️⃣ **Naïve Bayes (MultinomialNB)**  

Both models were trained on the processed dataset using **scikit-learn**.  

---

## **Evaluation**  
The models were evaluated using:  
- **Accuracy**  
- **Precision**  
- **Recall**  
- **F1-score**  
- **Confusion Matrix** (Visualized using Seaborn)  

---

## **Results**  

| Model               | Accuracy | Precision | Recall | F1-Score |
|---------------------|----------|------------|--------|----------|
| Logistic Regression | **87.15%**  | High       | High   | High     |
| Naïve Bayes        | **76.55%**  | Moderate   | Moderate | Moderate  |

✔ **Logistic Regression outperformed Naïve Bayes**, achieving higher accuracy and better classification results.  

📊 Confusion matrices were plotted using **Seaborn heatmaps** to analyze misclassifications.  

---

## **Installation**  

### **1️⃣ Clone the Repository**  
```bash
git clone https://github.com/OmBhutkar/text-classification.git
cd text-classification
```

### **2️⃣ Install Dependencies**  
```bash
pip install -r requirements.txt
```

### **3️⃣ Run the Script**  
```bash
python train_model.py
```

## **Usage**  
- Upload your dataset as `training.csv` and `test.csv`.  
- Modify preprocessing steps in `preprocessing.py` if needed.  
- Run `train_model.py` to train and evaluate models.  
- View results, including accuracy and confusion matrix visualizations.  

---

## **License**  
📜 This project is licensed under the **MIT License**.  
