import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from flask import Flask, render_template, request, redirect, send_from_directory
import pickle as pkl
import warnings
warnings.filterwarnings("ignore")


app = Flask(__name__)

@app.route("/")
def home():

    return render_template("home.html")


@app.route('/storage/<path:x>')
def storage(x):
    
    return send_from_directory("storage", x)


@app.route("/predict", methods = ["POST", "GET"])
def predict():

    return render_template("predict.html")


@app.route("/predict_result", methods = ["POST", "GET"])
def get_prediction_result():
    
    if request.method == "POST":
        a = request.form
        age = int(a["age"])
        if a["sex"] == "Male":
            sex = 1
        else:
            sex = 0
        height = float(a["height"])
        weight = float(a["weight"])
        bmi = float(weight/((height/100)**2))
        children = int(a["children"])
        if a["smoking"] == "Yes":
            smoker = 1
        else:
            smoker = 0
        if a["region"] == "Northeast":
            northeast = 1
            northwest = 0
            southeast = 0
        elif a["region"] == "Northwest":
            northeast = 0
            northwest = 1
            southeast = 0
        elif a["region"] == "Southeast":
            northeast = 0
            northwest = 0
            southeast = 1
        elif a["region"] == "Southwest":
            northeast = 0
            northwest = 0
            southeast = 0

        with open("Dfnya.pkl", "rb") as f:
            dfini = pkl.load(f)

        with open("Scaler.pkl", "rb") as f:
            scalertrf = pkl.load(f)

        x = dfini[["Age" ,"Sex", "BMI", "Children", "Smoker", "Northeast", "Northwest", "Southeast"]]
        y = dfini.Charges

        xtr, xts, ytr, yts = train_test_split(x, y, random_state = 0, test_size = 0.2)

        xts = xts
        yts = yts

        ranforht = RandomForestRegressor( bootstrap = True, criterion = "mse", max_depth = 50, 
                                          max_features = "auto", min_samples_leaf = 10, 
                                          min_samples_split = 7, n_estimators = 1200, random_state = 42
                                        )

        ranforht.fit(xtr, ytr)

        data = { "age" : [age], "sex" : [sex], "bmi" : [bmi], "children" : [children], "smoker" : [smoker], 
                 "northeast" : [northeast], "northwest" : [northwest], "southeast" : [southeast]
               }

        dfdata = pd.DataFrame.from_dict(data)

        z = scalertrf.transform(dfdata[["age", "bmi"]])
        zage = z[:,0]
        zbmi = z[:,1]
        dfdata["age"] = zage
        dfdata["bmi"] = zbmi

        predictive = int(ranforht.predict(dfdata[["age", "sex", "bmi", "children", "smoker", "northeast", "northwest", "southeast"]]))
        
        return render_template("predict_result.html", prediction = predictive)


@app.route("/visualization", methods = ["POST", "GET"])
def visualization():

    return render_template("visualization.html")


if __name__ == '__main__':
    app.run(debug = True, port = 7080)