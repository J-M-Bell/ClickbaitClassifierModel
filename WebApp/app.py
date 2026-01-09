from flask import Flask, render_template, request
import joblib
from Data.clickbait_classifier_model import ClickbaitClassifierModel

app = Flask(__name__)

model = joblib.load("./WebApp/Data/finalized_model.pkl") # Update path as needed

@app.route('/')
def index():
    """
    A function that renders the pages 
    """
    return render_template("index.html")

@app.route('/submit', methods=['POST'])
def predict():
    """
    A function that finds the prediction from
    the model and renders the page to display that
    prediction
    """
    if request.method == "POST":
        title = request.form.get("title")
        model = ClickbaitClassifierModel()
        prediction = model.predict(title)
        return render_template("index.html", prediction=prediction)

if __name__ == '__main__':
    app.run(port=3000, debug=True)