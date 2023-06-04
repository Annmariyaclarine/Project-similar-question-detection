from flask import Flask
from flask import render_template,request
from cust_funct import pred
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/check', methods=['POST'])
def check():
    # Get the input values from the form
    question1 = request.form['question1']
    question2 = request.form['question2']

    # Perform prediction using your deep learning model
    prediction = pred(question1, question2)

    # Return the result
    return render_template('prediction.html', prediction=prediction)


if __name__ == '__main__':
    app.run(debug=True)