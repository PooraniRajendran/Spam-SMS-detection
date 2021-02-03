from modeltraining import SMSModel
from flask import Flask,render_template,request


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
        smsModel = SMSModel()
        sms = request.form['sms']
        data = [sms]
        data_vector = smsModel.gettransform().transform(data).toarray()
        prediction = smsModel.getTrainedModel().predict(data_vector)
    return render_template('index.html',prediction=prediction)

@app.route('/train',methods=['POST'])
def train():
    if request.method == 'POST':
        smsModel = SMSModel()
        X,y = smsModel.preprocess()
        score = smsModel.trainModel(X,y)
        predictionText="accuracy score {}".format(score)
    return render_template('index.html', predictionText=predictionText)

if __name__ == '__main__':
    app.run(debug=True)