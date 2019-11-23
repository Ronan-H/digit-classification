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