# Import dependencies
import pandas as pd
import numpy as np
import pickle
import joblib as joblib
import json

# Load the dataset in a dataframe object and include only four features as mentioned
url = "http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/train.csv"
df = pd.read_csv(url)
include = ['Age', 'Sex', 'Embarked', 'Survived']  # Only four features
df_ = df[include]
#print column names
print(df_.columns.values)

# Data Preprocessing
categoricals = []
for col, col_type in df_.dtypes.iteritems():
    if col_type == 'O':
        categoricals.append(col)
    else:
        df_[col].fillna(0, inplace=True)

df_ohe = pd.get_dummies(df_, columns=categoricals, dummy_na=True)
print(df_ohe.columns.values)
# Logistic Regression classifier
# from sklearn.linear_model import LogisticRegression
import sklearn.linear_model as lm

dependent_variable = 'Survived'
x = df_ohe[df_ohe.columns.difference([dependent_variable])]
y = df_ohe[dependent_variable]
lr = lm.LogisticRegression()
lr.fit(x, y)

# Saving model to disk
pickle.dump(lr, open('model.pkl','wb'))
print("Model dumped!")

# Load the model that you just saved
lr = joblib.load('model.pkl')


model_columns = list(x.columns)
joblib.dump(model_columns, 'model_columns.pkl')
print("Models columns dumped!")
input = [{"Age":2,"Sex":"male","Embarked":"S"}]
print(type(input))
query =  pd.get_dummies(pd.DataFrame(input))
query = query.reindex(columns=model_columns, fill_value=0)
print(query)
print(lr.predict(query))

# lr = pickle.load(open('model.pkl','rb'))
# print(lr.predict([[1,0,21,8,6,200,0,1]])) #because of the splitting of the embarked columns and sex columns, we need to use joblib!!

# Saving the data columns from training
# model_columns = list(x.columns)
# joblib.dump(model_columns, 'model_columns.pkl')
# print("Models columns dumped!")
