from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load the trained model
model = pickle.load(open("RF.pkl", "rb"))

@app.route('/')
def index():
    return render_template('y_index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        pregnancies = float(request.form['pregnancies'])
        glucose = float(request.form['glucose'])
        blood_pressure = float(request.form['blood_pressure'])
        skin_thickness = float(request.form['skin_thickness'])
        insulin = float(request.form['insulin'])
        bmi = float(request.form['bmi'])
        diabetes_pedigree = float(request.form['diabetes_pedigree'])
        age = float(request.form['age'])

        user_input = [[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age]]
        prediction = model.predict(user_input)

        if prediction[0] == 1:
            result = "Likely to have diabetes"
        else:
            result = "Not likely to have diabetes"


        return render_template('y_result.html', result=result)

if __name__ == '__main__':
    app.run()
