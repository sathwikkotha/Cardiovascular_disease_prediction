# Cardiovascular_disease_prediction

# Heart Disease Prediction Using Logistic Regression

## Overview
This project implements a machine learning model using logistic regression to predict heart disease based on patient data. The dataset used contains medical attributes that help in identifying whether a person has heart disease.

## Dataset
The dataset is stored in a CSV file named `heart_disease_data.csv`. It includes several features such as:
- Age
- Sex
- Chest pain type
- Resting blood pressure
- Cholesterol level
- Fasting blood sugar
- Resting ECG results
- Maximum heart rate achieved
- Exercise-induced angina
- Oldpeak (ST depression induced by exercise relative to rest)
- Slope of the peak exercise ST segment
- Number of major vessels colored by fluoroscopy
- Thalassemia type
- Target variable (0 = No heart disease, 1 = Heart disease)

## Requirements
Ensure you have the following Python libraries installed:
```sh
pip install numpy pandas scikit-learn
```

## Implementation Steps
1. **Import necessary libraries:**
   - `numpy` for numerical operations
   - `pandas` for data handling
   - `sklearn.model_selection` for splitting data into training and test sets
   - `sklearn.linear_model` for logistic regression model
   - `sklearn.metrics` for evaluating model performance

2. **Load the dataset:**
   - Read the CSV file into a Pandas DataFrame.
   - Separate the features (`X`) from the target variable (`y`).

3. **Split the dataset:**
   - Use `train_test_split()` to divide the data into training (80%) and testing (20%) sets.

4. **Train the model:**
   - Fit a logistic regression model on the training data.

5. **Evaluate model accuracy:**
   - Predict outcomes on the training and test data.
   - Calculate accuracy scores.

6. **Make a prediction for a new sample:**
   - Convert input data into a NumPy array.
   - Reshape the array for model prediction.
   - Display the prediction result.

## Usage
Run the script in a Python environment:
```sh
python heart_disease_prediction.py
```
Example input for prediction:
```python
input_data = (41,0,1,130,204,0,0,172,0,1.4,2,0,2)
```
Output:
```
[1]
The person has heart disease
```

## Accuracy
- The model's accuracy on training and test datasets is displayed after execution.

## Future Enhancements
- Try different machine learning models like Decision Trees or Random Forests.
- Perform feature selection to improve accuracy.
- Deploy the model using Flask or FastAPI for real-world applications.

## License
This project is for educational purposes and is open for modifications and improvements.
