from flask import Flask, render_template, request
import joblib
from clickbait_model import ClickbaitClassifierModel

app = Flask(__name__)

model = joblib.load('finalized_model.pkl') # Update path as needed

@app.route('/', methods=['GET'])
def startPage():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    title = request.files['title']
    model = ClickbaitClassifierModel()
    prediction = ClickbaitClassifierModel.predict(title)
    


if __name__ == '__main__':
    app.run(port=3000, debug=True)