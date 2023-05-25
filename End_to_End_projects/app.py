import pickle
import os 
from flask import Flask,request,app ,jsonify,url_for,render_template
import numpy as np
import pandas as pd

app =Flask(__name__)
## load the model
regmodel=pickle.load(open("Reg_model.pkl",'rb'))
scaler=pickle.load(open("scaling.pkl",'rb'))

@app.route('/')
def home():
    return render_template('home.html')

# TO see on the postman with below api
@app.route('/predict_api',methods=['Get','POST'])
def predict_api():
    data=request.json['data']
    print(data)
    val=np.array(list(data.values())).reshape(1,-1)
    print(np.array(list(data.values())).reshape(1,-1))
    new_transform_data=scaler.transform(val)
    output=regmodel.predict(new_transform_data)
    # since output is in two demension as we see in the ipynb file
    print(output[0])
    return jsonify(output[0])

# to show on the website

@app.route('/predict',methods=['GET','POST'])
def predict():
    data=[float(x) for x in request.form.values()]
    final_input =scaler.transform(np.array(data).reshape(1,-1))
    print(final_input)
    output=regmodel.predict(final_input)[0]
    return render_template('home.html',prediction_text="The predicted house price is {}".format(output))



if __name__=="__main__":
    app.run(debug=True)

