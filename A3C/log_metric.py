import tensorflow as tf


class A3CMetric(tf.keras.metrics.Metric):

    def __init__(self, name='a3c_metric'):
        super(A3CMetric, self).__init__(name=name)
        self.loss = self.add_weight(name='loss', initializer='zeros')
        self.episode_step = self.add_weight(name='step', initializer='zeros')

    def update_state(self, y_true, y_pred=0, sample_weight=None):
        self.loss.assign_add(y_true)   # the y_tue is actuall los - it is not supposed to calculate loss itself
        self.episode_step.assign_add(1)

    def result(self):
        return self.loss/self.episode_step

    def reset_states(self):
        self.loss.assign(0)
        self.episode_step.assign(0)
