import tensorflow as tf
from tensorflow import keras

class LSTMModel(keras.Model):
    def __init__(self, input_dim, output_dim, lstm_units=64, dropout_rate=0.2):
        super(LSTMModel, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        
        self.input_layer = keras.layers.InputLayer(input_shape=(None, input_dim))
        self.lstm1 = keras.layers.LSTM(lstm_units, return_sequences=True)
        self.dropout1 = keras.layers.Dropout(dropout_rate)
        self.lstm2 = keras.layers.LSTM(lstm_units)
        self.dropout2 = keras.layers.Dropout(dropout_rate)
        self.dense = keras.layers.Dense(output_dim)

    def call(self, inputs):
        x = self.input_layer(inputs)
        x = self.lstm1(x)
        x = self.dropout1(x)
        x = self.lstm2(x)
        x = self.dropout2(x)
        return self.dense(x)

    def get_config(self):
        config = super().get_config()
        config.update({
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "lstm_units": self.lstm_units,
            "dropout_rate": self.dropout_rate
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

class TemporalAttention(keras.layers.Layer):
    def __init__(self, units):
        super(TemporalAttention, self).__init__()
        self.units = units
        self.W = keras.layers.Dense(units)
        self.V = keras.layers.Dense(1)

    def call(self, encoder_output):
        score = self.V(tf.nn.tanh(self.W(encoder_output)))
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = attention_weights * encoder_output
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, attention_weights

    def get_config(self):
        config = super().get_config()
        config.update({
            "units": self.units
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

class LSTMWithAttention(keras.layers.Layer):
    def __init__(self, input_dim, output_dim, lstm_units=64, dropout_rate=0.2, **kwargs):
        super(LSTMWithAttention, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        
        self.input_layer = keras.layers.InputLayer(input_shape=(None, input_dim))
        self.lstm = keras.layers.LSTM(lstm_units, return_sequences=True)
        self.attention = TemporalAttention(lstm_units)
        self.dropout = keras.layers.Dropout(dropout_rate)
        self.dense = keras.layers.Dense(output_dim)

    def call(self, inputs):
        x = self.input_layer(inputs)
        lstm_output = self.lstm(x)
        context_vector, _ = self.attention(lstm_output)
        x = self.dropout(context_vector)
        return self.dense(x)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "lstm_units": self.lstm_units,
            "dropout_rate": self.dropout_rate
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)