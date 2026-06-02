Air Quality Prediction Using Machine Learning
📌 Project Overview

This project focuses on analyzing and predicting air quality using the Air Quality UCI Dataset. The dataset contains various air pollutant measurements and environmental variables collected from sensors. Several Machine Learning algorithms are applied and compared to evaluate their performance in predicting air quality indicators.

📂 Dataset

The dataset used in this project is:

AirQualityUCI.csv
Contains air pollution measurements and sensor readings.
Features include:
Date
Time
CO(GT)
PT08.S1(CO)
NMHC(GT)
C6H6(GT)
PT08.S2(NMHC)
Temperature (T)
Relative Humidity (RH)
Absolute Humidity (AH)
Other sensor measurements
🛠 Technologies Used
Python
NumPy
Pandas
Matplotlib
Scikit-learn
Jupyter Notebook
📊 Data Preprocessing

The following preprocessing steps were performed:

Loaded the dataset using Pandas.
Checked for missing values.
Filled numerical missing values using the median.
Filled categorical missing values using the mode.
Removed unnecessary columns:
Unnamed: 15
Unnamed: 16
Applied Label Encoding to selected categorical features.
Visualized feature distributions using histograms.
🎯 Feature Selection
Target Variable
Y = data['CO(GT)']
Input Features

The following columns were removed from the feature set:

['CO(GT)', 'PT08.S1(CO)', 'NMHC(GT)', 'C6H6(GT)', 'PT08.S2(NMHC)']
🤖 Machine Learning Models Used
1. Support Vector Classifier (SVC)
SVC()
2. Random Forest Classifier
RandomForestClassifier()
3. Logistic Regression
LogisticRegression()
4. Decision Tree Classifier
DecisionTreeClassifier()
🔄 Train-Test Split
train_test_split(
    X,
    Y,
    test_size=0.3,
    random_state=42
)
Training Data: 70%
Testing Data: 30%
📈 Model Evaluation

The models were evaluated using:

accuracy_score()

Metrics reported:

Training Accuracy
Testing Accuracy
📋 Workflow
Import required libraries.
Load Air Quality dataset.
Handle missing values.
Encode categorical variables.
Perform exploratory data analysis.
Split data into training and testing sets.
Train multiple ML models.
Evaluate model performance.
Compare results.
🚀 How to Run
Clone the repository:
git clone https://github.com/yourusername/air-quality-prediction.git
Install dependencies:
pip install numpy pandas matplotlib scikit-learn jupyter
Open Jupyter Notebook:
jupyter notebook
Run:
AirQuality.ipynb
📁 Project Structure
AirQuality-Prediction/
│
├── AirQuality.ipynb
├── AirQualityUCI.csv
├── README.md
│
└── outputs/
    ├── visualizations
    └── model_results
