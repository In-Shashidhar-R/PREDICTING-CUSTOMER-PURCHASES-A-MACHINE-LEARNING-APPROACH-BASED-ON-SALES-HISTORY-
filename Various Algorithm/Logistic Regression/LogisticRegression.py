import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
# loading the csv data to a Pandas DataFrame
heart_data = pd.read_csv('C:\\Users\\INFINIX\\Desktop\\Study\\Semester - 4\\22BIO211 - BIO-2\\Project codes\\heart (1).csv')
# print first 5 rows of the dataset
heart_data.head()
# print last 5 rows of the dataset
heart_data.tail()
# number of rows and columns in the dataset
heart_data.shape
# getting some info about the data
heart_data.info()
# checking for missing values
heart_data.isnull().sum()
# statistical measures about the data
heart_data.describe()
# checking the distribution of Target Variable
heart_data['target'].value_counts()
X = heart_data.drop(columns='target', axis=1)
Y = heart_data['target']
print(X)
print(Y)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)
print(X.shape, X_train.shape, X_test.shape)
model = LogisticRegression()
# training the LogisticRegression model with Training data
model.fit(X_train, Y_train)
# accuracy on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print('Accuracy on Training data : ', training_data_accuracy)
# accuracy on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('Accuracy on Test data : ', test_data_accuracy)

import pickle
filename = 'heart_disease_model.sav'
pickle.dump(model, open(filename, 'wb'))
# loading the saved model
loaded_model = pickle.load(open('heart_disease_model.sav', 'rb'))
for column in X.columns:
  print(column)
import numpy as np
# Get input from the user
age = int(input("Enter the age: "))
sex = int(input("Enter the sex (0 for female, 1 for male): "))
cp = int(input("Enter the chest pain type (0 for typical angina, 1 for atypical angina, 2 for non-anginal pain, 3 for asymptomatic): "))
trestbps = int(input("Enter the resting blood pressure: "))
chol = int(input("Enter the cholesterol level: "))
fbs = int(input("Enter the fasting blood sugar level (0 for < 120 mg/dl, 1 for >= 120 mg/dl): "))
restecg = int(input("Enter the resting electrocardiographic results (0 for normal, 1 for having ST-T wave abnormality, 2 for showing probable or definite left ventricular hypertrophy): "))
thalach = int(input("Enter the maximum heart rate achieved: "))
exang = int(input("Enter the exercise induced angina (0 for no, 1 for yes): "))
oldpeak = float(input("Enter the ST depression induced by exercise relative to rest: "))
slope = int(input("Enter the slope of the peak exercise ST segment (0 for upsloping, 1 for flat, 2 for downsloping): "))
ca = int(input("Enter the number of major vessels colored by fluoroscopy: "))
thal = int(input("Enter the thallium stress test result (0 for fixed defect, 1 for normal, 2 for reversable defect, 3 for not reversable defect): "))


input_data = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]

# Convert the input data to a NumPy array
input_data_as_numpy_array = np.asarray(input_data)

# Reshape the NumPy array as we are predicting for only one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

# Load the saved model
loaded_model = pickle.load(open('heart_disease_model.sav', 'rb'))

# Predict the output
prediction = loaded_model.predict(input_data_reshaped)

# Print the prediction
if (prediction[0] == 0):
  print('The person does not have a heart disease')
else:
  print('The person has heart disease')