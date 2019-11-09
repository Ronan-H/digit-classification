from flask import Flask
from flask import request
import json
import numpy as np
import keras
from PIL import Image
app = Flask(__name__)


def load_model():
    json_file = open('../model/model.json', 'r')
    model_json = json_file.read()
    json_file.close()
    model = keras.models.model_from_json(model_json)
    # load weights into new model
    model.load_weights("../model/model.h5")
    print("Loaded model from disk")
    return model


model = load_model()


@app.route('/')
def hello():
    return app.send_static_file("digit-classifier.html")


@app.route("/classify", methods=["POST"])
def classify():
    pixel_data = request.form.getlist("pixelData[]")
    print(pixel_data)
    # convert boolean array to values from 0-255
    pixel_nums = np.array([0 if v == "true" else 255 for v in pixel_data], dtype=np.uint8)
    # model_input = pixel_nums.reshape((28, 28, 1))
    model_input = pixel_nums.reshape((150, 150))
    image = Image.fromarray(model_input)
    image.save("test_image.png")
    model_output = model.predict(model_input)
    print("PREDICTION: ", model_output)
    return json.dumps({'success': True}), 200, {'ContentType': 'application/json'}


if __name__ == '__main__':
    app.run()