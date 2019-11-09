from flask import Flask
from flask import request
app = Flask(__name__)


@app.route('/')
def hello():
    return app.send_static_file("digit-classifier.html")


@app.route("/classify", methods=["POST"])
def classify():
    print(request.form)
    return "Test"

if __name__ == '__main__':
    app.run()