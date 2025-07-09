# 💰 Medical Insurance Cost Prediction with Linear Regression

This project predicts **medical insurance costs** based on personal health and demographic data using a **Linear Regression** model.

## 📂 Dataset

- **Source**: [Medical Cost Personal Dataset](https://www.kaggle.com/datasets/mirichoi0218/insurance)
- **File**: `insurance.csv`
- **Features**:
  - `age`: Age of primary beneficiary
  - `sex`: Gender (male/female)
  - `bmi`: Body Mass Index
  - `children`: Number of dependents covered by insurance
  - `smoker`: Smoking status (yes/no)
  - `region`: Residential area in the US (southeast, southwest, northeast, northwest)
  - `charges`: Final insurance cost (target variable)

## ⚙️ Technologies Used

- Python 3.x
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn

## 🔍 Exploratory Data Analysis (EDA)

The project performs several visual analyses:
- Age, BMI, and Charges distributions
- Count plots of categorical features (sex, smoker, region, children)
- Checks for null/missing values and dataset shape
- Descriptive statistics for all columns

## 🧹 Data Preprocessing

- Missing values: Not present in this dataset
- Categorical encoding:
  - `sex`: male → 0, female → 1
  - `smoker`: yes → 0, no → 1
  - `region`: southeast → 0, southwest → 1, northeast → 2, northwest → 3

## 📈 Model Training

A `LinearRegression` model is trained using scikit-learn:

0. Training Code
   ```python
   from sklearn.linear_model import LinearRegression
   regressor = LinearRegression()
   regressor.fit(X_train, Y_train)
   test_size: 20%
   random_state: 2


📊 Evaluation
- Metric: R² Score
- Training R² and Test R² are calculated
- Visual comparison between actual vs predicted charges using scatter plots


🔮 Prediction Example
 1. Code
    ```python
    input_data = (31, 1, 25.74, 0, 1, 0)  # Example: age, sex, bmi, children, smoker, region
    input_data_reshaped = np.asarray(input_data).reshape(1, -1)
    prediction = regressor.predict(input_data_reshaped)
    print("Predicted insurance cost: $", prediction[0])


🚀 How to Run

1. Clone this repository:
   ```bash
   git clone https://github.com/erenntorun/medical-insurance-cost-prediction-with-linear-regression
   
2. Navigate to the project folder:
   ```bash
   cd medical-insurance-cost-prediction-with-linear-regression

3. Install dependencies:
   ```bash
   pip install numpy pandas scikit-learn matplotlib seaborn

4. Make sure your dataset path is correct in the script.

5. Run the script:
   ```bash
   python Medical_Insurance_Cost_Prediction.py


📌 Notes
Linear regression is a simple and interpretable model, but other models (e.g., XGBoost, RandomForest) may provide better performance for this dataset.

No feature scaling is applied since the model handles numerical values directly.

Visualizations can help identify outliers and nonlinear trends.


Created by @eren 👨‍💻
