from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)
lr = pickle.load(open('model.pkl', 'rb'))
cv = pickle.load(open('cv.pkl', 'rb'))

@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html')



@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        text=request.form['text']
        normalized=cv.transform([text])
        result=lr.predict(normalized)
        print(result)
        if(result==1):
            return render_template('index.html',prediction_text="Positive Review")
        else:
            return render_template('index.html',prediction_text="Negative Review")
    #else:
     #   return render_template('index.html',prediction_texts="Positive Review")

if __name__=="__main__":
    app.run()
