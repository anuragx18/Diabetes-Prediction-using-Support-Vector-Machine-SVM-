# Diabetes Prediction using Support Vector Machine (SVM)

## 📌 Description
This repository contains a machine learning project that predicts whether a person has **diabetes** or not based on several diagnostic features.  
The project uses a **Support Vector Machine (SVM)** classifier, a widely used algorithm for classification tasks, to build the predictive model.  

The entire process, from data handling to model evaluation, is clearly documented in a **Jupyter Notebook**.

---

## 🚀 Project Highlights
The notebook provides a comprehensive look at the following steps:

- **Libraries**: Importing machine learning libraries including `numpy`, `pandas`, and `sklearn`.  
- **Data Collection & Analysis**:  
  - Loading the `diabetes.csv` dataset.  
  - Inspecting rows, dataset shape, and statistical summaries.  
- **Data Preprocessing**:  
  - Separating features (`X`) and target variable (`Y`).  
  - Standardizing features with `StandardScaler` to ensure equal contribution.  
  - Splitting data into **training (80%)** and **testing (20%)** sets using `train_test_split` with stratification.  
- **Model Training**:  
  - Training a **Linear SVM** classifier (`svm.SVC(kernel='linear')`) on the training set.  
- **Model Evaluation**:  
  - Assessing performance using **accuracy score** on training data.  

---

## 🛠️ Technologies Used
- **Python** – Core programming language  
- **Jupyter Notebook / Google Colab** – Development environment  
- **Pandas** – Data loading and manipulation  
- **NumPy** – Numerical computations and array handling  
- **Scikit-learn (Sklearn)** – Machine learning library  
  - `StandardScaler` for preprocessing  
  - `train_test_split` for splitting data  
  - `svm.SVC` for SVM classification  

---

## ▶️ How to Run the Notebook
You can run this project easily in **Google Colab** or Jupyter Notebook:

1. **Environment**: Open in [Google Colab](https://colab.research.google.com/) or Jupyter Notebook.  
2. **Dataset**: Ensure `diabetes.csv` is available in your Colab environment or in the same local directory as the notebook.  
3. **Execution**: Open `2-Diabetes-Prediction-SVM.ipynb` and run all cells sequentially.  

---

## 🔮 Next Steps & Possible Improvements
- ✅ Evaluate the model on **test data** for realistic performance metrics.  
- ✅ Implement **cross-validation** for more robust accuracy estimation.  
- ✅ Experiment with different **SVM kernels** (`rbf`, `poly`) and tune hyperparameters.  
- ✅ Handle or impute features with **zero values** (e.g., `BloodPressure`, `BMI`) for improved accuracy.  

---

## 👨‍💻 Author
- **Name**: Anurag Lende  
- **LinkedIn**: [[LinkedIn Profile](https://www.linkedin.com/in/anuraglende/)] 

---
