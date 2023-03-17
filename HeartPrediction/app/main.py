from sklearn.svm import SVC
import pandas as pd
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import plotly.offline as py
import plotly.graph_objs as go
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from flask import Flask, render_template, request, url_for, flash, redirect

app = Flask(__name__)
app.config['SECRET_KEY'] = 'harsh004'


@app.route('/')
def index():
    return render_template('index.html')

"""
    Original file is located at
        https://colab.research.google.com/drive/1KBGb5Gvznl3RYxZuR0Jzn1l8jfAXMTB4

    https://www.kaggle.com/code/microvision/heart-disease-exploratory-data-analysis/notebook
"""


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        age = request.form['age']
        sex = request.form.get("sex")
        cpt = request.form.get['chest_pain_type']
        r_ecg = request.form.get['resting_ecg']
        mhr = request.form['max_heart_rate']
        st_depr = request.form['st_depression']
        st_slope = request.form.get['st_slope']
        passed_data = svm(age, sex, cpt, r_ecg, mhr, st_depr, st_slope)
        return render_template('predict.html', condition=passed_data)
    
    return render_template('predict.html')


def svm(a, s, c, r, m, sd, ss):
    sns.set(style='darkgrid')

    df = pd.read_csv(
        'https://raw.githubusercontent.com/OliMations/uoeHeartDiseasePrediction/main/heart.csv')
    df

    df.describe()


    df.columns = ['Age', 'Sex', 'Chest_pain_type', 'Resting_bp',
                'Cholesterol', 'Fasting_bs', 'Resting_ecg',
                'Max_heart_rate', 'Exercise_induced_angina',
                'ST_depression', 'ST_slope', 'Num_major_vessels',
                'Thallium_test', 'Condition']
    df.head(10)

    df.shape

    X = X = df[['Age', 'Sex', 'Chest_pain_type',
                'Resting_ecg', 'Max_heart_rate',
                'ST_depression', 'ST_slope',]]
    y = df.Condition

    X_train, X_test, y_train,  y_test = train_test_split(
        X, y, test_size=0.33, random_state=42)

    clf = SVC(kernel='linear') 
    clf.fit(X_train, y_train)
    
    # i= 247
    
    age = a
    sex = s
    cpt = c
    r_ecg = r
    mhr = m
    st_depr = sd
    st_slope = ss

    arr = np.array([age, sex, cpt, r_ecg, mhr, st_depr, st_slope]).reshape(1, -1)
    usr_data = pd.DataFrame(arr, columns=['Age', 'Sex', 'Chest_pain_type', 'Resting_ecg', 'Max_heart_rate', 'ST_depression', 'ST_slope'])
   
    # user_data = np.array(chest_pain_type, st_slope, max_heart_rate, resting_ecg)
    # fitting x samples and y classes
    
    y_pred = clf.predict(usr_data)
    print(y_pred)
    # accuracy = metrics.accuracy_score(y_test, y_pred)*100
    # print(f"Accuracy: {accuracy}")
    pred = y_pred[0]
    return pred
    # xg boost try
