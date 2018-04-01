import numpy as np
import pandas as pd
import pickle as pickle
from keras.preprocessing.text import Tokenizer
from keras.models import load_model
import tensorflow as tf


class Predictor(object):
    def __init__(self):

        self.model = load_model('../models/model_v1.1_sigmoid.h5')
        self.graph = tf.get_default_graph()
        self.labelmap = {
            0:'business',
            1:'entertainment',
            2:'technology',
            3:'health'
        }
        with open('../models/tokenizer_v1.1_sigmoid.pickle', 'rb') as handle:
            self.tokenizer = pickle.load(handle)

    def predict(self, text):

        print(text)
        x = self.tokenizer.texts_to_matrix([text])
        print(x)
        with self.graph.as_default():
            pred = self.model.predict(np.array(x)).argmax()
        print(pred)
        output = self.labelmap[pred]
        print(output)
        return output
