import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'  # kill warning about tensorflow
import tensorflow as tf
import numpy as np
import sys

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import load_model


class TrainModel:
    def __init__(self, num_layers, width, batch_size, learning_rate, input_dim, output_dim):
        self._input_dim = input_dim
        self._output_dim = output_dim
        self._batch_size = batch_size
        self._learning_rate = learning_rate
        self._model = self._build_model(num_layers, width)


    def _build_model(self, num_layers, width):
        """
        Build and compile a Transformer-based Deep Q-Network.
        The Transformer encoder learns inter-lane/intersection relationships.
        Dense layers downstream are kept small for the prototype stage.
        Scale num_layers/width up when moving to 100+ intersections.
        """
        inputs = keras.Input(shape=(self._input_dim,))
        
        # Reshape to (Batch, Sequence, Features)
        # We treat each lane-group as a token with 10 features (cells)
        num_lanes = self._input_dim // 10
        x = layers.Reshape((num_lanes, 10))(inputs)
        
        # Transformer Encoder Block
        # 1. Self Attention — learns which lane-groups should coordinate
        attention_output = layers.MultiHeadAttention(num_heads=2, key_dim=10)(x, x)
        x = layers.Add()([x, attention_output])
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        
        # 2. Feed Forward Network
        x_ff = layers.Dense(width, activation='relu')(x)
        x_ff = layers.Dense(10)(x_ff) # Project back to feature dim
        x = layers.Add()([x, x_ff])
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        
        # Flatten and downstream Dense layers
        x = layers.Flatten()(x)
        
        for _ in range(num_layers):
            x = layers.Dense(width, activation='relu')(x)
            x = layers.Dropout(0.1)(x)  # Light regularization to reduce overfitting
            
        outputs = layers.Dense(self._output_dim, activation='linear')(x)

        model = keras.Model(inputs=inputs, outputs=outputs, name='transformer_model')
        model.compile(loss='mse', optimizer=Adam(learning_rate=self._learning_rate))
        return model
    

    def predict_one(self, state):
        """
        Predict the action values from a single state
        """
        state = np.reshape(state, [1, self._input_dim])
        return self._model.predict(state, verbose=0)


    def predict_batch(self, states):
        """
        Predict the action values from a batch of states
        """
        return self._model.predict(states, verbose=0)


    def train_batch(self, states, q_sa):
        """
        Train the nn using the updated q-values
        """
        self._model.fit(states, q_sa, epochs=1, verbose=0)


    def save_model(self, path):
        """
        Save the current model in the folder as h5 file and a model architecture summary as png
        """
        self._model.save(os.path.join(path, 'trained_model.h5'))
        # plot_model(self._model, to_file=os.path.join(path, 'model_structure.png'), show_shapes=True, show_layer_names=True)


    @property
    def input_dim(self):
        return self._input_dim


    @property
    def output_dim(self):
        return self._output_dim


    @property
    def batch_size(self):
        return self._batch_size


class TestModel:
    def __init__(self, input_dim, model_path):
        self._input_dim = input_dim
        self._model = self._load_my_model(model_path)


    def _load_my_model(self, model_folder_path):
        """
        Load the model stored in the folder specified by the model number, if it exists
        """
        model_file_path = os.path.join(model_folder_path, 'trained_model.h5')
        
        if os.path.isfile(model_file_path):
            loaded_model = load_model(model_file_path, compile=False)
            return loaded_model
        else:
            sys.exit("Model number not found")


    def predict_one(self, state, verbose=0):
        """
        Predict the action values from a single state
        """
        state = np.reshape(state, [1, self._input_dim])
        return self._model.predict(state, verbose=verbose)


    @property
    def input_dim(self):
        return self._input_dim