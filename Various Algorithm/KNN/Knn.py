import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import pickle

heart_data = pd.read_csv('C:\\Users\\INFINIX\\Desktop\\Study\\Semester - 4\\22BIO211 - BIO-2\\Project codes\\heart (1).csv')

X = heart_data.drop(columns='target', axis=1)
Y = heart_data['target']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

model = KNeighborsClassifier(n_neighbors=3)  

model.fit(X_train, Y_train)

X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print('Accuracy on Training data: ', training_data_accuracy)

X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('Accuracy on Test data: ', test_data_accuracy)


filename = 'heart_disease_model_knn.sav'
pickle.dump(model, open(filename, 'wb'))

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
thal = int(input("Enter the thallium stress test result (0 for fixed defect, 1 for normal, 2 for reversible defect, 3 for not reversible defect): "))

input_data = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]

input_data_as_numpy_array = np.asarray(input_data)

input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

loaded_model_knn = pickle.load(open('heart_disease_model_knn.sav', 'rb'))

prediction_knn = loaded_model_knn.predict(input_data_reshaped)

if prediction_knn[0] == 0:
    print('The person does not have a heart disease')
else:
    print('The person has heart disease')
