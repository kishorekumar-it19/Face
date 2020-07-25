import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import pickle
app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
@app.route('/')
def home():
    return render_template('index.html')
@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    ws=float(request.form['windspeed'])
    di=request.form['wind direction']
    if ws==5 and di=='north':
        return render_template('index.html', prediction_text='The power to be generated is {} Mega Watts'.format(ws),a=di)

if __name__ == "__main__":
    app.run()

