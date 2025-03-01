from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load Model and Preprocessing Tools
model = pickle.load(open("models/best_model.pkl", "rb"))
scaler = pickle.load(open("models/scaler.pkl", "rb"))
imputer = pickle.load(open("models/imputer.pkl", "rb"))
encoder_holiday = pickle.load(open("models/encoder_holiday.pkl", "rb"))
encoder_weather = pickle.load(open("models/encoder_weather.pkl", "rb"))

# Holiday and Weather Options
HOLIDAYS = [
    "None", "Columbus Day", "Veterans Day", "Thanksgiving Day", "Christmas Day",
    "New Years Day", "Washingtons Birthday", "Memorial Day", "Independence Day",
    "State Fair", "Labor Day", "Martin Luther King Jr Day"
]
WEATHER = ["Clouds", "Clear", "Rain", "Drizzle", "Mist", "Haze", "Fog", "Thunderstorm", "Snow", "Squall", "Smoke"]

@app.route('/')
def home():
    return render_template('index.html', holidays=HOLIDAYS, weather=WEATHER)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = [
            request.form['holiday'],
            float(request.form['temp']),
            float(request.form['rain']),
            float(request.form['snow']),
            request.form['weather']
        ]

        # Encode categorical values safely
        data[0] = encoder_holiday.transform([data[0]])[0] if data[0] in encoder_holiday.classes_ else -1
        data[4] = encoder_weather.transform([data[4]])[0] if data[4] in encoder_weather.classes_ else -1

        # Convert to numpy array, reshape, impute, and scale
        data = np.array(data).reshape(1, -1)
        data = imputer.transform(data)
        data = scaler.transform(data)

        prediction = model.predict(data)[0]
        return render_template('index.html', prediction_text=f'Predicted Traffic Volume: {round(prediction)}', holidays=HOLIDAYS, weather=WEATHER)

    except Exception as e:
        return render_template('index.html', error_text=f'Error: {str(e)}', holidays=HOLIDAYS, weather=WEATHER)

if __name__ == '__main__':
    app.run(debug=True)
