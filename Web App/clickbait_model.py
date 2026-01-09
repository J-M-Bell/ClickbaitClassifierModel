import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import BernoulliNB
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import joblib
import pickle

class ClickbaitClassifierModel():
    """
    A class to instantiate the Clickbait Classifier ML Model
    """
    def __init__(self):
        """
        constructor for ClickbaitClassifierModel
        
        :param self: ClickbaitClassifierModel - The ClickbaitClassifierModel object
        """

        # Get data to be processed
        X, y = self._getData()

        # Create vectorizer and process data
        self.tdidf_vectorizer = TfidfVectorizer(tokenizer=self._custom_tokenizer, token_pattern=None)
        X_vectorized = self.tdidf_vectorizer.fit_transform(X)

        # Create and fit model
        self.model = BernoulliNB(alpha=0.1).fit(X_vectorized, y)

     
    def _getData(self):
        """
        A method that gets the clickbait and non-clickbait titles
        to be processed
        
        :param self: ClickbaitClassifierModel - The ClickbaitClassifierModel object
        """
        
        # Formatting text files and mapping them to a dictionary
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

        # Create DataFrame to hold data
        df = pd.DataFrame(columns=["title", "target"])
        df = pd.DataFrame(data_dict)
        data = df.sample(frac=1, random_state=1).reset_index(drop=True)

        # Create feature and target variables
        X = data["title"]
        y = data["target"]
        return X, y

    def _custom_tokenizer(self, text):
        """
        A method to process text into vectors
        
        :param self: ClickbaitClassifierModel - The ClickbaitClassifierModel object
        :param text: String - clickbait message string

        :return an array of words
        """

        # Get stopwords function and Word Lemmatizer
        en_stopwords = stopwords.words('english')
        lemmatizer = WordNetLemmatizer()

        # Process text
        new_text = text.lower() #lowercase
        new_text = re.sub(r"([^\w\s])", "", new_text) #remove punctuation
        new_text = word_tokenize(new_text) #tokenize
        for word in new_text: #remove stopwords
            if word in en_stopwords:
                new_text.remove(word)
        new_text = [lemmatizer.lemmatize(token) for token in new_text] #lemmatize

        return new_text
    
    def _createPickleFile(self):
        """
        A method to create the deploy the ml model
        as a pickle file
        
        :param self: ClickbaitClassifierModel - The ClickbaitClassifierModel object
        """
        filename = 'finalized_model.pkl'
        with open(filename, 'wb') as file:
            pickle.dump(self.model, file) # Serialize and save the model

    def predict(self, text):
        """
        A method that calls the ml model's predict method 
        
        :param self: ClickbaitClassifierModel - The ClickbaitClassifierModel object
        :param text: String - clickbait message string from user

        :return the prediction as a String
        """

        #Transform text
        vectorized_output = self.tdidf_vectorizer.transform([text])

        #Get prediction
        prediction = self.model.predict(vectorized_output)
        prediction = prediction[0]
        return prediction
    
    def _loadModel(self):
        """
        A helper function to load the model as a pickle file
        
        :param self: ClickbaitClassifierModel - The ClickbaitClassifierModel object
        """
        model = joblib.load('my_model.pkl')


# Tests the model class
# model = ClickbaitClassifierModel()
# prediction = model.predict("Which TV Female Friend Group Do You Belong In")
# print(prediction)
