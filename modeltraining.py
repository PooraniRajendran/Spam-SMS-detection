from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import pandas as pd
import pickle


class SMSModel:
    def __init__(self):
        pass

    def preprocess(self):
        df = pd.read_csv("Updated_Spam_file.csv", encoding="latin-1")
        df['label'] = df['class'].map({'ham': 0, 'spam': 1})
        X = df['messages']
        y = df['label']
        cv = CountVectorizer()
        X = cv.fit_transform(X)
        filename='datatransform.pkl'
        pickle.dump(cv,open(filename,'wb'))
        return X,y

    def trainModel(self,X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
        nbclassifier = MultinomialNB()
        nbclassifier.fit(X_train,y_train)
        score = nbclassifier.score(X_test,y_test)

        filename = "spamclassifier.pkl"
        pickle.dump(nbclassifier,open(filename,'wb'))

        return score

    def gettransform(self):
        cv = pickle.load(open('datatransform.pkl', 'rb'))
        return cv

    def getTrainedModel(self):
        nbclassifier = pickle.load(open('spamclassifier.pkl', 'rb'))
        return nbclassifier