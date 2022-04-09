from src.__refactored__.utils.configs import cfg
import tensorflow as tf
from abc import ABCMeta, abstractmethod

class ModelTemplate(metaclass=ABCMeta):
    def __init__(self, scope, dataset):
        self.scope = scope
        self.global_step = tf.compat.v1.get_variable('global_step', shape=[], dtype=tf.int32,
                                           initializer=tf.constant_initializer(0), trainable=False)

        #---------------------------------------------------------------------------------------------------------------
        #                                       parameters from Config
        #---------------------------------------------------------------------------------------------------------------

        #-----------------------------------------------
        # control - these are just for driving the process
        #-----------------------------------------------
        self.model_type = cfg.model                         # default='tesa', help='tesa, vanila_sa, or cbow'
        self.data_src = cfg.data_source                     # default='mimic3', help='mimic3 or cms'
        self.gpu_device = '/gpu:' + str(cfg.gpu)            # cfg.gpu -> default=0
        self.verbose = cfg.verbose                          # default=False, help='print ...'

        #-----------------------------------------------
        # Hierarchical TeSa
        #-----------------------------------------------
        # self.is_plus_sa = cfg.is_plus_sa                    # default=True, help='add multi-dim self-attention'
        # self.is_plus_date = cfg.is_plus_date                # default=True, help='add temporal interval'
        # self.predict_type = cfg.predict_type                # default='dx', help='dx:diagnosis; re:readmission,death: mortality, los: length of stay'

        #-----------------------------------------------
        # training
        #-----------------------------------------------
        self.is_date_encoding = cfg.is_date_encoding        # default=False, help='To control date encoding'
        self.is_scale = cfg.is_scale                        # default=True, help='to scale the attention facts'
        self.activation = cfg.activation                    # default='relu', help='activation function'
        self.max_epoch = cfg.max_epoch                      # default=20, help='Max Epoch Number'
        self.num_samples = cfg.num_samples                  # default=5, help='Number of negative examples to sample'

        #-----------------------------------------------
        # code Processing
        #-----------------------------------------------
        self.embedding_size = cfg.embedding_size            # default=100, help='code embedding size'

        #-----------------------------------------------
        # validatation
        #-----------------------------------------------
        # self.valid_size (not set or referenced here)      # default=1000, help='evaluate similarity size'
        self.valid_samples = cfg.valid_examples             # list(range(1,self.valid_size+1))  # Only pick dev samples in the head of the distribution.
        self.top_k = cfg.top_k                              # default=1, help='number of nearest neighbors'

        #-----------------------------------------------
        # Hierarchical Self-Attention for prediction
        #-----------------------------------------------
        self.hierarchical = cfg.hierarchical                # default=True, help='hierarchical attention'

        #---------------------------------------------------------------------------------------------------------------
        #                                       parameters from dataset
        #---------------------------------------------------------------------------------------------------------------
        self.vocabulary_size = len(dataset.dictionary)
        self.dates_size = dataset.days_size
        self.reverse_dict = dataset.reverse_dictionary

        # ------ start ------
        self.tensor_dict = {}
        self.loss = None
        self.optimizer = None
        self.accuracy = None
        self.summary = None
        self.opt = None
        self.train_op = None

    @abstractmethod
    def build_network(self):
        pass

    @abstractmethod
    def build_loss_optimizer(self):
        pass

    @abstractmethod
    def build_accuracy(self):
        pass

