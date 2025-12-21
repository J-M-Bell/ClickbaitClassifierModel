from flask import Flask, render_template, request
import joblib
from clickbait_model import ClickbaitClassifierModel

app = Flask(__name__)

model = joblib.load('finalized_model.pkl') # Update path as needed

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/submit', methods=['POST'])
def predict():
    if request.method == "POST":
        title = request.form.get("title")
        model = ClickbaitClassifierModel()
        prediction = model.predict(title)
        return render_template("index.html", prediction=prediction)

if __name__ == '__main__':
    app.run(port=3000, debug=True)