import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

HIDDEN_UNITS_1 = 256  # Number of units in the first hidden layer
HIDDEN_UNITS_2 = 128  # Number of units in the second hidden layer


class ActorCriticNet:
    def __init__(self, num_actions, num_states):
        self.num_actions = num_actions
        self.num_states = num_states

    def create_network(self):
        inputs = layers.Input(shape=(self.num_states,))
        hidden_1 = layers.Dense(units=HIDDEN_UNITS_1, activation='relu')(inputs)
        hidden_2 = layers.Dense(units=HIDDEN_UNITS_2, activation='relu')(hidden_1)
        actor = layers.Dense(units=self.num_actions, activation='softmax')(hidden_2)
        critic = layers.Dense(units=1)(hidden_2)

        model = keras.Model(inputs, [actor, critic])
        return model
