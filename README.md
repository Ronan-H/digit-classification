# Digit Classification

## What this is
This is my 4th year *emerging technologies* assignment. There are two main parts: a [Jupyter Notebook](https://jupyter.org/) which trains a neural network on the [MNIST dataset of handwritten digits](http://yann.lecun.com/exdb/mnist/) step by step using [Keras](https://keras.io/), in order to later classify digits drawn by a user using a simple web applicatoin written using [Flask](https://www.fullstackpython.com/flask.html).

## Notable files in this repo

* **/model**: files relating to the Keras model (neural network)
  * **model.ipynb**: Jupyter Notebook; building, training, and evaluating the neural network.
  * **model.h5**: neural network model serialized to a file (for use in the Flask web application)


* **/flask**: files relating to the Flask web application
  * **app.py**: entry point to the web application
  * **image_ops.py**: image operations (decoding from POST request, "MNISTifying", image to model input, etc)
  * **static/**: static resources (main page, styling)
  

## How to run the web application

*Prerequisites: Python 3, TensorFlow, Keras, Flask. Install using [pip](https://pypi.org/project/pip/) and/or [Anaconda](https://www.anaconda.com/distribution/)*

Run **python3 app.py** from the *flask* directory.

(or **python app.py**, depending on your installation)

The TensorFlow warnings can safely be ignored.

Visit **http://127.0.0.1:5000/** in your web browser to view the web application.

## Using the web application

Draw a digit on the canvas using your mouse. When you release left click, the digit you drew should be classified automatically and displayed below the canvas. Clicking "Clear" erases the whole canvas.

## How to make changes and run the model in Jupyter Notebook

*Prerequisites: Python 3, TensorFlow, Keras, Flask. Install using [pip](https://pypi.org/project/pip/) and/or [Anaconda](https://www.anaconda.com/distribution/)*

Run **jupyter lab** from the *flask* directory.

From here you can edit and re-run the notebook (which will also output a new **model.h5** file)

# How the model works

You can read the [Jupyter Nobtebook](https://github.com/Ronan-H/digit-classification/blob/master/model/model.ipynb) to see how the model works, which has code separated out into cells, with some markdown inbetween explaining what each part does.

# "MNISTifying" user drawn digits

Users draw a digit on a large (eg. 200x200) canvas, and the model classifies 28x28 digits, so the image has to be at least be resized before being fed into the model. But if we copy the way the digits were resized in the MNIST dataset...


> The original black and white (bilevel) images from NIST were size normalized to fit in a 20x20 pixel box while preserving their aspect ratio. The resulting images contain grey levels as a result of the anti-aliasing technique used by the normalization algorithm. the images were centered in a 28x28 image by computing the center of mass of the pixels, and translating the image so as to position this point at the center of the 28x28 field.

> (from http://yann.lecun.com/exdb/mnist/)

...the predictions should be much more accurate, so that's what I did. I broke this down into 4 steps:

1. Crop image to fit digit (with no border)
2. Extend image out into a square (width, height become whichever is bigger)
3. Scale the image down to 20x20, and expand edges out to 28x28 (image gets aliased while doing this)
4. Center digit within image based on Center of Mass

Doing this produces images that are more or less indistinguishable from digits from the MNIST dataset.

# Canvas image encoding

The easiest way to send the canvas data up to the web server is to use *canvas.toDataURL()*, producing a PNG encoded image, sent up as Base64 text. I decided to make my own encoding, taking advantage of the fact that the canvas pixels are always either transparent, or black (appearing to the user as white or black). 

The first encoding goes through the image pixel by pixel, counting up how many white/black pixels there are in a row. Once the value changes from black to white or white to black, this value is recorded in an array. I later found out that this is called a [Run-length encoding](https://en.wikipedia.org/wiki/Run-length_encoding).
