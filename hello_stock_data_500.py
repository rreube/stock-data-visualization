from flask import Flask,request, g
import pandas as pd
import csv
import sqlite3

from flask import Flask, request, g
from flask_cors import CORS


app = Flask(__name__)

DATABASE = '../stock_data_500.db'

app.config.from_object(__name__)
CORS(app)

def connect_to_database():
    return sqlite3.connect(app.config['DATABASE'])

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"


@app.route("/stock_data/<string:ticker>")
def get_stock_data(ticker):
    con = sqlite3.connect(DATABASE)
    return pd.read_sql(f"""SELECT * FROM stock_fundamental_data
                            WHERE ticker = '{ticker}'""",con).to_json(orient='records')

@app.route("/stock_data_tech/<string:ticker>")
def get_stock_data_technical(ticker):
    con = sqlite3.connect(DATABASE)
    return pd.read_sql(f"""SELECT * FROM stock_technical_data
                            WHERE ticker = '{ticker}'""",con).to_json(orient='records')

@app.route("/top_ten_list/")
def get_top10():
    con = sqlite3.connect(DATABASE)
    return pd.read_sql(f"""SELECT * FROM top_ten_list""", con).to_json(orient='records')

if __name__ == '__main__':
  app.run()
