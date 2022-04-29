from src.mortality_prediction.configs import cfg
import tensorflow as tf
from abc import ABCMeta, abstractmethod

class ModelTemplate(metaclass=ABCMeta):
    def __init__(self, scope, modelParams):
        self.scope = scope
        self.global_step = tf.compat.v1.get_variable('global_step', shape=[], dtype=tf.int32,
                                           initializer=tf.constant_initializer(0), trainable=False)

        # ------ start ------
        self.tensor_dict = {}
        self.loss = None
        self.optimizer = None
        self.accuracy = None
        self.summary = None
        self.opt = None
        self.train_op = None
        self.vect_file = modelParams["vect_file"]
        self.hidden_units = modelParams["hidden_units"]

    @abstractmethod
    def build_network(self):
        pass

    @abstractmethod
    def build_loss_optimizer(self):
        pass

    @abstractmethod
    def build_accuracy(self):
        pass

