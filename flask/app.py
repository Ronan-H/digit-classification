from flask import Flask
from flask import request
import json
app = Flask(__name__)


@app.route('/')
def hello():
    return app.send_static_file("digit-classifier.html")


@app.route("/classify", methods=["POST"])
def classify():
    print(request.form.getlist("pixelData[]"))
    return json.dumps({'success': True}), 200, {'ContentType': 'application/json'}

if __name__ == '__main__':
    app.run()