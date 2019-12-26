from flask import Flask, render_template, request
import pymysql

app = Flask(__name__)

@app.route('/')
def hello():
    return "Hello!"

@app.route('/input')
def input():
    return render_template('input.html')

@app.route('/meta', methods = ['POST'])
def meta():
    val1 = request.form['code']
    val2 = request.form['hyhg']

    conn = pymysql.connect(
        host='13.209.87.99', port=3306, user='root',
        passwd='mypassword', db='testdb', charset='utf8'
    )
    conn.query('set character_set_server=utf8;')
    try:
        with conn.cursor() as cursor:
            sql = "SELECT * FROM meta_table WHERE hyhg=%s AND course_code LIKE %s"
            cursor.execute(sql, (val2, val1+'%'))
            result = cursor.fetchall()
            print(result)
    finally:
        conn.close()
    val3 = result[0][4]
    result_list = list(result)
    result_list.insert(0, ['hyhg', 'course_code', 'course_title', 'credit', 'instructor', 'time',
        'room', 'quota', 'participants', 'major_quota', 'second_major', 'grade_1', 'grade_2',
        'grade_3', 'grade_4', 'exchange_student', 'max_mileage', 'min_result', 'max_result',
        'average'])
    result = tuple(result_list)


    return render_template('meta.html', result=None, resultData=result, resultUPDATE=None)

if __name__ == '__main__':
    app.run()
