import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

df = pd.read_csv (r'C:\Users\makis\OneDrive\Υπολογιστής\Data Mining\healthcare-dataset-stroke-data\healthcare-dataset-stroke-data.csv')

features = ['gender','age','hypertension','heart_disease','ever_married','work_type','Residence_type','avg_glucose_level','bmi']

df = df.drop("smoking_status", axis=1)
#mapping all categorical columns to numeric
df["gender"] = df['gender'].map({'Female':0, 'Male':1, 'Other':2})
df["ever_married"] = df['ever_married'].map({'Yes':0, 'No':1})
df["work_type"] = df['work_type'].map({"children":0, "Govt_job":1, "Never_worked":2, "Private":3, "Self-employed":4})
df["Residence_type"] = df['Residence_type'].map({"Rural":0, "Urban":1})

df = df.interpolate(method='linear', limit_direction='both') 

#split the dataframe
X_train, X_test, y_train, y_test = train_test_split(df[features], df['stroke'], test_size=0.25, random_state=100)
#implementing RandomForest
rf = RandomForestClassifier(n_estimators=5,  random_state=50)
rf.fit(X_train, y_train)
y_pred=rf.predict(X_test)
#creating confusion matrix
confusion_matrix = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
sns.heatmap(confusion_matrix, annot=True)
prec_score = metrics.precision_score(y_test, y_pred)
f1_score = metrics.f1_score(y_test, y_pred)
rec_score = metrics.recall_score(y_test, y_pred)
plt.title("Linear Regression Method")
plt.show()
# prediction = rf.predict([[0,70,0,0,0,3,0,117,1]]) 
# prediction = rf.predict([[1,35,0,1,0,1,1,117,1]]) 
# print ('Predicted Result: ', prediction)