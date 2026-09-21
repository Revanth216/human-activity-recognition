# Human Activity Recognition with K-Means Feature Selection

A machine learning project for **Human Activity Recognition (HAR)** using smartphone sensor data. This project investigates whether **K-Means-based unsupervised feature selection** can reduce the dimensionality of the sensor feature space while maintaining or improving classification performance and reducing Gaussian Naive Bayes execution time.

## 📌 Project Overview

Human Activity Recognition involves identifying physical activities such as walking, sitting, standing, and walking upstairs from smartphone sensor measurements.

The UCI Human Activity Recognition Using Smartphones dataset contains **561 engineered sensor features** derived from smartphone accelerometer and gyroscope measurements.

This project explores an alternative to using all 561 features:

> **Can similar sensor features be grouped using K-Means, with representative features selected from each cluster, to create a smaller and more computationally efficient classification model?**

The project compares a **Gaussian Naive Bayes baseline using all 561 features** against models trained using K-Means-selected feature subsets.

---

## 🎯 Objectives

* Establish a baseline Human Activity Recognition model using all 561 features.
* Apply **K-Means clustering to the feature space** rather than clustering observations.
* Select representative features from each K-Means cluster.
* Evaluate feature subsets ranging from **10 to 100 features**.
* Compare classification accuracy and model execution time.
* Analyze the trade-off between dimensionality reduction and predictive performance.

---

## 🧠 Methodology

### 1. Data Acquisition

The project downloads the Human Activity Recognition Using Smartphones dataset from the UCI Machine Learning Repository.

The dataset contains smartphone sensor measurements associated with six human activities.

### 2. Data Preprocessing

The preprocessing pipeline includes:

* Label encoding of activity classes
* Stratified train/test splitting
* Standardization using `StandardScaler`
* Fitting the scaler only on the training data to avoid direct test-set leakage

### 3. Baseline Model

A **Gaussian Naive Bayes** classifier is trained using all:

```text
561 features
```

The baseline establishes the reference point for accuracy and execution time.

### 4. K-Means Feature Selection

Instead of clustering individual observations, the feature matrix is transposed so that each feature becomes a data point.

```text
Original matrix:

Samples × Features
    ↓
    ↓ transpose
Features × Samples
```

K-Means is then applied to group similar features.

For each cluster:

1. Identify the features belonging to the cluster.
2. Calculate the distance between each feature and the cluster centroid.
3. Select the feature closest to the centroid.
4. Use that feature as the representative feature for the cluster.

This produces smaller feature sets containing original sensor features rather than transformed dimensions.

### 5. Feature Budgets

The experiment evaluates:

```text
10
20
30
40
50
60
70
80
90
100
```

selected features.

---

## 📊 Results

The experiment produced the following key results:

| Metric                     | Baseline | K-Means Reduced |
| -------------------------- | -------: | --------------: |
| Features                   |      561 |              50 |
| Accuracy                   |   73.15% |          78.59% |
| Feature Reduction          |        — |            ~91% |
| Gaussian NB Execution Time | Baseline |     ~92% lower* |

### Accuracy Improvement

The 50-feature configuration improved the observed Gaussian Naive Bayes accuracy from:

```text
73.15% → 78.59%
```

This corresponds to an improvement of:

```text
+5.44 percentage points
```

while reducing the feature space from:

```text
561 → 50 features
```

or approximately:

```text
91% fewer features
```

### Computational Efficiency

The reduced model showed approximately **92% lower Gaussian Naive Bayes execution time** in the recorded experiment.

> **Important:** this timing refers to the Gaussian Naive Bayes model execution measured in the experiment. K-Means feature-selection time is measured separately and should not be interpreted as a 92% reduction in total end-to-end pipeline runtime.

---

## 📈 Experiments

The project evaluates the relationship between:

* Number of selected features
* Classification accuracy
* Gaussian Naive Bayes execution time
* K-Means feature-selection time

This allows the experiment to examine the **accuracy–efficiency trade-off** rather than evaluating only a single reduced feature set.

### Accuracy Comparison

The project generates a plot comparing:

```text
K-Means reduced-feature accuracy
vs.
561-feature baseline accuracy
```

### Computational Efficiency

A second visualization compares:

```text
Gaussian Naive Bayes execution time
vs.
number of selected features
```

---

## 🛠️ Technologies

* **Python**
* **NumPy**
* **Pandas**
* **Scikit-learn**
* **Matplotlib**
* **Requests**
* **UCI Machine Learning Repository**

### Machine Learning

* K-Means Clustering
* Gaussian Naive Bayes
* Feature Selection
* Standardization
* Classification Evaluation

---

## 📂 Project Structure

```text
human-activity-recognition/
│
├── human_activity_recognition.py
├── human_activity_recognition.ipynb
├── kmeans_naive_bayes_results.csv
└── README.md
```

---

## 🚀 How to Run

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/human-activity-recognition.git
cd human-activity-recognition
```

### 2. Install dependencies

```bash
pip install numpy pandas scikit-learn matplotlib requests
```

### 3. Run the Python script

```bash
python human_activity_recognition.py
```

The script will:

1. Download the UCI HAR dataset.
2. Load and preprocess the data.
3. Train the 561-feature Gaussian Naive Bayes baseline.
4. Perform K-Means feature selection.
5. Evaluate 10–100 selected features.
6. Generate comparison plots.
7. Export experimental results to:

```text
kmeans_naive_bayes_results.csv
```

---

## 🔬 Key Findings

The experiment demonstrates that reducing the feature space does not necessarily require sacrificing classification performance.

The observed 50-feature configuration:

* Reduced the feature space by approximately **91%**.
* Improved observed accuracy from **73.15% to 78.59%**.
* Reduced Gaussian Naive Bayes execution time by approximately **92%** in the recorded experiment.

The results suggest that clustering correlated or similar sensor features and retaining representative variables can provide a compact feature representation for Human Activity Recognition.

---

## ⚠️ Experimental Notes

The reported results correspond to the specific experimental configuration and random seed used in the implementation.

For a more rigorous research evaluation, future experiments can include:

* Using the **official UCI train/test subject split**
* Repeating experiments across multiple random seeds
* Reporting **mean ± standard deviation**
* Adding macro-F1 and weighted-F1 metrics
* Comparing K-Means feature selection against PCA and other feature-selection techniques
* Measuring complete end-to-end inference and preprocessing latency

These extensions would provide a stronger assessment of generalization and computational efficiency.

---

## 📚 Dataset

**Human Activity Recognition Using Smartphones**

UCI Machine Learning Repository

Dataset ID: 240

The dataset contains recordings from smartphone accelerometer and gyroscope sensors collected from human subjects performing six different activities.

---

## 👤 Author

**Revanth Valupadasu**


---

## ⭐ Project Highlights

```text
561 → 50 features
~91% feature reduction

73.15% → 78.59% accuracy
+5.44 percentage points

~92% lower Gaussian NB execution time
```
