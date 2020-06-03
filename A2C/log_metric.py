import tensorflow as tf


class A2CMetric(tf.keras.metrics.Metric):

    def __init__(self, name='a2c_metric'):
        super(A2CMetric, self).__init__(name=name)
        self.loss = self.add_weight(name='tp', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.loss.assign_add(tf.square(y_true - y_pred))

    def result(self):
        return self.loss

    def reset_states(self):
        self.loss.assign(0)
