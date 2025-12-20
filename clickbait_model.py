import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import ConfusionMatrixDisplay, precision_score, recall_score, PrecisionRecallDisplay, make_scorer
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve, GridSearchCV, validation_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.pipeline import Pipeline
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import joblib
import pickle

class ClickbaitClassifierModel():
    def __init__(self):
        self.tdidf_vectorizer = TfidfVectorizer(tokenizer=self._custom_tokenizer, token_pattern=None)
        X, y = self._data_processing()
        X_vectorized = self.tdidf_vectorizer.fit_transform(X)
        self.model = BernoulliNB(alpha=0.1).fit(X_vectorized, y)

     
    def _data_processing(self):
        df = pd.DataFrame(columns=["title", "target"])

        # Formatting text files
        titles = []
        targets = []
        file_path_dict = {'clickbait': './clickbait_data.txt', 'non clickbait': './non_clickbait_data.txt'}
        for key, value in file_path_dict.items():
            with open(value, 'r') as file:
                for line_number, line in enumerate(file):
                    line = line.strip()
                    if line != "":
                        titles.append(line)
                        targets.append(key)
        data_dict = {"title": titles, "target": targets}
        df = pd.DataFrame(data_dict)

        data = df.sample(frac=1, random_state=1).reset_index(drop=True)

        X = data["title"]
        y = data["target"]
        return X, y

    def _custom_tokenizer(self, text):
        en_stopwords = stopwords.words('english')
        lemmatizer = WordNetLemmatizer()
        new_text = text.lower() #lowercase

        new_text = re.sub(r"([^\w\s])", "", new_text) #remove punctuation

        for word in new_text.split(): #remove stopwords
            if word in en_stopwords:
                new_text = new_text.replace(word, "")

        new_text = word_tokenize(new_text) #tokenize

        new_text = [lemmatizer.lemmatize(token) for token in new_text] #lemmatize
        return new_text
    
    def createPickleFile(self):
        filename = 'finalized_model.pkl'
        with open(filename, 'wb') as file:
            pickle.dump(self.model, file) # Serialize and save the model

    def predict(self, text):
        vectorized_output = self.tdidf_vectorizer.transform([text])
        prediction = self.model.predict(vectorized_output)
        prediction = prediction[0]
        return prediction
    
    # def loadModel(self):
    #     model = joblib.load('my_model.pkl')







# joblib.dump(model, 'my_model.pkl')
