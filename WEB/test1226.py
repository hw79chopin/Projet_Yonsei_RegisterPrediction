import pymysql
import pandas as pd
import numpy as np
from flask import Flask, render_template, request
from xgboost import XGBClassifier, plot_importance
from sklearn.model_selection import train_test_split
from sklearn.metrics import auc, f1_score, accuracy_score
import matplotlib.pyplot as plt

app = Flask(__name__)

@app.route('/')
def hello():
    return "Hello!"

@app.route('/input2')
def input():
    return render_template('input2.html')

@app.route('/meta2', methods = ['POST'])
def meta():
    val1 = request.form['course_code']
    val2 = request.form['instructor']
    val3 = int(request.form['quota'])
    val4 = int(request.form['max_mileage'])
    val5 = int(request.form['mileage'])
    val6 = int(request.form['major'])
    val7 = int(request.form['double_major'])
    val8 = int(request.form['major_quota'])
    val9 = int(request.form['second_major'])
    val10 = int(request.form['enrolled_courses'])
    val11 = int(request.form['graduation'])
    val12 = int(request.form['first_enroll'])
    val13 = float(request.form['credits_rate'])
    val14 = float(request.form['previous_credits_rate'])
    val15 = int(request.form['grade'])
    val16 = int(request.form['grade_quota_1'])
    val17 = int(request.form['grade_quota_2'])
    val18 = int(request.form['grade_quota_3'])
    val19 = int(request.form['grade_quota_4'])

    conn = pymysql.connect(
        host='13.209.87.99', port=3306, user='root',
        passwd='mypassword', db='testdb', charset='utf8'
    )

    try:
        with conn.cursor() as cursor:
            sql = "SELECT * FROM df_pred WHERE course_code=%s AND instructor=%s"
            cursor.execute(sql, (val1, val2))
            result = cursor.fetchall()
    finally:
        conn.close()

    data = [[val3, val4, val5, val8, val9, val16, val17, val18, val19, val10, val11, val6, val7, val12, val13, val14, 0, 0, 0, 0]]

    new_df = pd.DataFrame(result, columns = ['enrolled', 'hyhg', 'course_code', 'instructor', 'quota',
        'max_mileage','mileage','major_quota', 'second_major','grade_quota_1',
        'grade_quota_2', 'grade_quota_3','grade_quota_4','enrolled_courses', 'graduation',
        'major','double_major','first_enroll','credits_rate','previous_credits_rate',
        'grade_1','grade_2','grade_3','grade_4'])

    df_input = pd.DataFrame(data, columns = ['quota','max_mileage','mileage','major_quota','second_major',
        'grade_quota_1','grade_quota_2','grade_quota_3','grade_quota_4','enrolled_courses',
        'graduation', 'major','double_major','first_enroll','credits_rate',
        'previous_credits_rate','grade_1','grade_2','grade_3','grade_4'])

    if val15 == 1:
        df_input['grade_1'] = df_input['grade_1'].replace(0,1)
    elif val15 == 2:
        df_input['grade_2'] = df_input['grade_2'].replace(0,1)
    elif val15 == 3:
        df_input['grade_3'] = df_input['grade_3'].replace(0,1)
    elif val15 == 4:
        df_input['grade_4'] = df_input['grade_4'].replace(0,1)

    if val8 == -1:
            df_input['major_quota'] = df_input['major_quota'].replace(-1, np.nan)

    df_input.loc[(df_input['grade_quota_1'] ==-1) &
         (df_input['grade_quota_2'] ==-1) &
         (df_input['grade_quota_3'] ==-1) &
         (df_input['grade_quota_4'] ==-1), ['grade_quota_1', 'grade_quota_2',
                              'grade_quota_3', 'grade_quota_4']] = np.nan

    X_train = new_df[list(new_df)[4:]]
    y_train = new_df['enrolled']
    X_test = df_input

    model = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=0.9, gamma=0,
              learning_rate=0.02, max_delta_step=0, max_depth=20,
              min_child_weight=1, missing=None, n_estimators=150, n_jobs=1,
              objective='binary:logistic', random_state=0,
              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=0,
              silent=0, subsample=0.9, verbosity=1)

    model.fit(X_train, y_train)
    prediction = round(model.predict_proba(X_test)[0][1], 3)

    optimum_mileage = val4 + 1
    passornot = []

    for i in range(val4, 0, -1):
        optimum_mileage -= 1
        df_temp = df_input.copy()
        df_temp['mileage'] = df_temp['mileage'].replace(val5,i)

        prediction_2 = model.predict_proba(df_temp)
        passornot.append([prediction_2, optimum_mileage])
        mileage_prop = optimum_mileage + 1
        if prediction_2[0] == 0:
            break
        elif prediction_2[0] == 1:
            pass


    return render_template('meta2.html', result=None, resultData=((prediction, mileage_prop),), resultUPDATE=None)

if __name__ == '__main__':
    app.run()
