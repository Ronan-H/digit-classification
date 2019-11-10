from flask import Flask
from flask import request
import json
import numpy as np
import keras
from PIL import Image
app = Flask(__name__)


def load_model():
    model = keras.models.load_model("../model/model.h5")
    print("Loaded model from disk")
    return model


model = load_model()


@app.route('/')
def hello():
    return app.send_static_file("digit-classifier.html")


@app.route("/classify", methods=["POST"])
def classify():
    pixel_data = request.form.getlist("pixelData[]")
    # convert boolean array to values from 0-255
    pixel_nums = np.array([255 if v == "true" else 0 for v in pixel_data], dtype=np.uint8)
    model_input = pixel_nums.reshape((250, 250))
    image = Image.fromarray(model_input).resize([28, 28], Image.NEAREST)
    model_input = np.array(image).reshape((1, 28, 28, 1))
    image.save("test_image.png")
    model_output = model.predict_classes(model_input)
    return str(model_output[0])


if __name__ == '__main__':
    app.run(threaded=False)
