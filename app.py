from flask import Flask, url_for, request, redirect, Blueprint
from flask import render_template
import model

app = Flask(__name__)
@app.route('/')
# @app.route('/')
def hello():
    return render_template('review.html')

@app.route('/result', methods=['POST', 'GET'])
def result():
    if request.method == 'POST':
        predictlinearReg, predictBayesRidge, predictSVMModel = model.predict(request.form['review'])
        # result = {'SVM score': predicSVM,
        #           'Gaussian Bayes score': predictGaussianBayes}
        result = {'linear regression score': predictlinearReg,
                  'bayes ridge score': predictBayesRidge,
                  'svm score': predictSVMModel}
        return render_template("result.html", result=result)


if __name__ == '__main__':
    app.run()
