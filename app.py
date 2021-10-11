import numpy as np
import pickle
from flask import Flask,request,render_template

app=Flask(__name__)

gender_pred=pickle.load(open('gender_pred.pkl','rb'))

@app.route('/')   ## Home page
def Home_page():
    return render_template('gender1.html')

@app.route('/sub',methods =['POST'])
def predict():
     if request.method == 'POST':
        m = request.form["Mathscore"]
        r = request.form['Readingscore']
        w = request.form['writingscore']
        
        
        data = np.array([[m,r,w]])
        data=data.reshape(1,-1)
        my_prediction = gender_pred.predict(data)
        
        return render_template('result.html', prediction=my_prediction)

if __name__=='__main__':
         app.run(debug=True)

        
        