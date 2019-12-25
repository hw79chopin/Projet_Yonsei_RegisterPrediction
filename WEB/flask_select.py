from flask import Flask
from flask import render_template
import pymysql.cursors

app = Flask(__name__)

@app.route('/')
def index():
    return "<h1>index</h1>"

@app.route('/select/', methods=['GET'])
def select():
    conn = pymysql.connect(
        host='13.125.236.201', port=3306, user='root',
        passwd='mypassword', db='newdb', charset='utf8'
    )
    try:
        with conn.cursor() as cursor:
            sql = "SELECT enrolled, mileage FROM table_a LIMIT 30"
            cursor.execute(sql)
            result = cursor.fetchall()
            print(result)
    finally:
        conn.close()
    return render_template('dbdata.html', result=None,
            resultData=result[0], resultUPDATE=None)
