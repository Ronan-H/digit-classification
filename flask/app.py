from flask import Flask
from flask import request
import json
import numpy as np
import keras
from PIL import Image
from PIL import ImageOps
app = Flask(__name__)


def load_model():
    model = keras.models.load_model("../model/model.h5")
    print("Loaded model from disk")
    return model


model = load_model()


def scale_and_center(img, img_size, scaled_size):
    immat = img.load()
    crop_left = img_size
    crop_right = 0
    crop_up = img_size
    crop_down = 0
    for x in range(img_size):
        for y in range(img_size):
            if immat[(x, y)] != 0:
                crop_left = min(x, crop_left)
                crop_right = max(x, crop_right)
                crop_up = min(y, crop_up)
                crop_down = max(y, crop_down)

    cropped = img.crop((crop_left, crop_up, crop_right + 1, crop_down + 1))
    cropped.save("cropped.png")
    if cropped.size[0] > cropped.size[1]:
        expand_amount = (cropped.size[0] - cropped.size[1]) // 2
        cropped = ImageOps.expand(cropped, border=(0, expand_amount, 0, expand_amount))
    else:
        expand_amount = (cropped.size[1] - cropped.size[0]) // 2
        cropped = ImageOps.expand(cropped, border=(expand_amount, 0, expand_amount, 0))

    scaled = cropped.resize([20, 20], Image.NEAREST)
    cropped.save("resized-20.png")
    scaled = ImageOps.expand(scaled, border=4)

    # https://stackoverflow.com/questions/37519238/python-find-center-of-object-in-an-image
    immat = scaled.load()
    m = np.zeros((scaled_size, scaled_size))

    for x in range(scaled_size):
        for y in range(scaled_size):
            m[x, y] = immat[(x, y)] != 0

    s = np.sum(np.sum(m))

    if s == 0:
        # canvas is blank, do nothing
        return scaled

    m = m / s

    # marginal distributions
    dx = np.sum(m, 1)
    dy = np.sum(m, 0)

    # expected values
    cx = np.sum(dx * np.arange(scaled_size))
    cy = np.sum(dy * np.arange(scaled_size))

    middle = scaled_size / 2
    offset_x = cx - middle
    offset_y = cy - middle

    # https://stackoverflow.com/questions/37584977/translate-image-using-pil
    a = 1
    b = 0
    c = round(offset_x)  # left/right (i.e. 5/-5)
    d = 0
    e = 1
    f = round(offset_y)  # up/down (i.e. 5/-5)
    translated = scaled.transform(scaled.size, Image.AFFINE, (a, b, c, d, e, f))

    return translated


@app.route('/')
def hello():
    return app.send_static_file("digit-classifier.html")


@app.route("/classify", methods=["POST"])
def classify():
    scaled_size = 28

    img_size = int(request.values.get("width"))
    pixel_data = request.form.getlist("pixelData[]")
    # convert boolean array to values from 0-255
    pixel_nums = np.array([255 if v == "true" else 0 for v in pixel_data], dtype=np.uint8)
    model_input = pixel_nums.reshape((img_size, img_size))
    # image = Image.fromarray(model_input).resize([28, 28], Image.NEAREST)
    image = Image.fromarray(model_input)
    image = scale_and_center(image, img_size, scaled_size)
    model_input = np.array(image).reshape((1, scaled_size, scaled_size, 1))
    image.save("test_image.png")
    model_output = model.predict_classes(model_input)
    return str(model_output[0])


if __name__ == '__main__':
    app.run(threaded=False)
