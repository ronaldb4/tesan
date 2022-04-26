from src.concept_embedding.configs import cfg
import tensorflow as tf
from abc import ABCMeta, abstractmethod

class ModelTemplate(metaclass=ABCMeta):
    def __init__(self, scope, dataset):
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

        self.activation = None
        self.is_scale = None

        self.valid_samples = cfg.evaluation["valid_examples"]                   # list(range(1,self.valid_size+1))  # Only pick dev samples in the head of the distribution.
        self.embedding_size = cfg.modelParams["embedding_size"]                 # default=100, help='code embedding size'
        self.num_negative_examples = cfg.modelParams["num_negative_examples"]   # default=5, help='Number of negative examples to sample'

        if "activation" in cfg.modelParams:
            self.activation = cfg.modelParams["activation"]                    # default='relu', help='activation function'
        if "is_scale" in cfg.modelParams:
            self.is_scale = cfg.modelParams["is_scale"]                        # default=True, help='to scale the attention facts'

        #---------------------------------------------------------------------------------------------------------------
        #                                       parameters from dataset
        #---------------------------------------------------------------------------------------------------------------
        self.vocabulary_size = len(dataset.dictionary)
        self.dates_size = dataset.days_size
        self.reverse_dict = dataset.reverse_dictionary

    @abstractmethod
    def build_network(self):
        pass

    @abstractmethod
    def build_loss_optimizer(self):
        pass

    @abstractmethod
    def build_accuracy(self):
        pass

