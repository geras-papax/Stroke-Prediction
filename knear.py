import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

sns.set()

df = pd.read_csv (r'C:\Users\makis\OneDrive\Υπολογιστής\Data Mining\healthcare-dataset-stroke-data\healthcare-dataset-stroke-data.csv')

features = ['gender','age','hypertension','heart_disease','ever_married','work_type','Residence_type','avg_glucose_level','smoking_status']

df = df.drop("bmi", axis=1)#drop numerical nan column
#mapping all categorical columns to numeric
df["gender"] = df['gender'].map({'Female':0, 'Male':1, 'Other':2})
df["ever_married"] = df['ever_married'].map({'Yes':0, 'No':1})
df["work_type"] = df['work_type'].map({"children":0, "Govt_job":1, "Never_worked":2, "Private":3, "Self-employed":4})
df["Residence_type"] = df['Residence_type'].map({"Rural":0, "Urban":1})
df["smoking_status"] = df['smoking_status'].map({"formerly smoked":0, "smokes":1, "never smoked":2, "Unknown":np.nan})

#implementing KNN for 3 neighbors
imputer = KNNImputer(n_neighbors=3)
dfn = imputer.fit_transform(df)
#reforming the dataframe
dfn = pd.DataFrame(dfn, columns= df.columns).astype(df.dtypes.to_dict())
#split the dataframe
X_train, X_test, y_train, y_test = train_test_split(dfn[features], dfn['stroke'], test_size=0.25, random_state=100)
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
plt.title("KNN Method")
plt.show()
# prediction = rf.predict([[1,35,0,1,0,1,1,180,1]]) 
# print ('Predicted Result: ', prediction)