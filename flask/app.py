from flask import Flask
from flask import request
import keras

import image_ops

app = Flask(__name__)
# load model from file
model = keras.models.load_model("../model/model.h5")


@app.route('/')
def hello():
    """
    Serves a page to draw digits on and classify them
    """

    return app.send_static_file("digit-classifier.html")


@app.route("/classify", methods=["POST"])
def classify():
    """
    Classifies a drawn digit based on pixel data taken from the canvas
    :return: Digit classification (0-9)
    """

    img_size = int(request.values.get("size"))
    image_data = request.values.get("pixelData")

    # convert post data to PIL Image
    canvas_img = image_ops.post_data_to_image(image_data, img_size)
    # crop, resize, and translate the drawn digit to match the MNIST data set as closely as possible
    mnistified_img = image_ops.mnistify_image(canvas_img, save_stages=False)
    # convert the image to a form that the model accepts as input
    model_input = image_ops.image_to_model_input(mnistified_img)

    # make and return the digit classification
    model_output = model.predict_classes(model_input)
    return str(model_output[0])


if __name__ == '__main__':
    app.run(host="0.0.0.0", threaded=False)
