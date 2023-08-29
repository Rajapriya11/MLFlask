import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import pickle
from collections import Counter
from imblearn.over_sampling import SMOTE
data = pd.read_csv("D:/MLFlask/diabetes_prediction.csv")
#Preprocessing
# Drop rows with other in the "gender" column
values_to_drop = ['Other']  
data = data.query('gender not in @values_to_drop')
print(data)

# Convert the "age" column to numeric, replacing non-numeric values with NaN
data['age'] = pd.to_numeric(data['age'], errors='coerce')

# Filter out rows with NaN values and non-integer values
data = data[data['age'].notnull()]
data = data[data['age'] % 1 == 0]  # Keep only rows with integer values

# Convert the "age" column to integers
data['age'] = data['age'].astype(int)

# Print the cleaned DataFrame
print(data)
#Label Encoding
le = LabelEncoder()
le.fit(data['gender'])
data['gender'] = le.transform(data['gender'])
le.fit(data['smoking_history'])
data['smoking_history'] = le.transform(data['smoking_history'])
#Seperating independent and dependent columns
x = data[["gender", "age", "hypertension", "heart_disease", "smoking_history", "bmi","HbA1c_level","blood_glucose_level"]]
y = data['diabetes']
#train and test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
# Print class distribution before applying SMOTE
print("Class distribution before SMOTE:", Counter(y_train))

# Apply SMOTE to the training data
smote = SMOTE(random_state=42)
x_train_resampled, y_train_resampled = smote.fit_resample(x_train, y_train)

# Print class distribution after applying SMOTE
print("Class distribution after SMOTE:", Counter(y_train_resampled))


#Model training
RF=RandomForestClassifier()
RF.fit(x_train_resampled, y_train_resampled)
#Model testing
predictions = RF.predict(x_test)
RF.score(x_test,y_test)
print("Accuracy:",RF.score(x_test,y_test)*100)
#Save the model
file='diab.sav'
pickle.dump(RF,open(file,'wb'))
