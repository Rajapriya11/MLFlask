from flask import Flask,request,jsonify,render_template
import pickle
app=Flask(__name__)
#load the model
model=pickle.load(open('diab.sav','rb'))

@app.route('/')
def Home():
    result=''
    return render_template('index.html',**locals())
@app.route('/predict',methods=['POST','GET'])
def predict():
    if request.method == 'POST':
      gender = int(request.form["gender"])
      age = int(request.form["age"])
      hypertension = int(request.form["hypertension"])
      heart_disease = int(request.form["heart_disease"])
      smoking_history = int(request.form["smoking_history"])
      bmi = float(request.form["bmi"])
      HbA1c_level = float(request.form["HbA1c_level"])
      blood_glucose_level= int(request.form["blood_glucose_level"])
    # get prediction
    input_cols = [[gender,age, hypertension,heart_disease, smoking_history,bmi, HbA1c_level,blood_glucose_level]]
    prediction = model.predict(input_cols)
    return render_template("index.html",prediction_text=prediction[0],**locals())



if __name__ =='__main__':
    app.run(debug=True)
