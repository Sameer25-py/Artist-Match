from flask import Flask,render_template,request,redirect,url_for
from models.predictor import predictor 

app = Flask(__name__)

artists={}

@app.route('/')
def results():
    return render_template('index.html')
@app.route('/about')
def about():
	return render_template('about.html')

@app.route('/', methods=['POST'])
def my_form_post():
    text = request.form['text']
    if text == "":
    	return render_template('index.html')
    pred = predictor()
    artists=pred.predict(text)
    return render_template('results.html',artists=artists)

if __name__ == '__main__':
	app.run(debug=True)