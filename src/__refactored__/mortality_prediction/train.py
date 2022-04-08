import tensorflow as tf
import numpy as np
from os.path import join

from src.__refactored__.concept_embedding.models.cbow_model import CBOWModel
from src.__refactored__.mortality_prediction.models.glove_model import GloveModel
from src.__refactored__.mortality_prediction.models.mce_model import MCEModel
from src.__refactored__.mortality_prediction.models.med2vec_model import Med2VecModel
from src.__refactored__.mortality_prediction.models.sg_model import SGModel
from src.__refactored__.mortality_prediction.models.raw_model import RawModel
from src.__refactored__.mortality_prediction.models.tesa_model import TesaModel
from src.__refactored__.mortality_prediction.models.delta_model import DeltaModel
from src.__refactored__.mortality_prediction.models.sa_model import SAModel
from src.__refactored__.mortality_prediction.models.normal_model import NormalModel
from src.__refactored__.utils.configs import cfg

from src.__refactored__.utils.graph_handler import GraphHandler
from src.__refactored__.utils.record_log import RecordLog
from src.__refactored__.mortality_prediction.data.dataset import MortalityDataset
from src.__refactored__.mortality_prediction.evaluation.__evaluation_template__ import EvaluationTemplate as et

import warnings
warnings.filterwarnings('ignore')
logging = RecordLog()


def train():

    if cfg.gpu_mem is None:
        gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=cfg.gpu_mem, allow_growth=True)
        graph_config = tf.compat.v1.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)

    else:
        gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=cfg.gpu_mem)
        graph_config = tf.compat.v1.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)

    visit_threshold = 1
    num_steps = cfg.num_steps
    data_set = MortalityDataset()
    data_set.prepare_data(visit_threshold)
    data_set.build_dictionary()
    data_set.load_data()
    sample_batches = data_set.generate_batch(num_steps)

    all_batches = data_set.generate_batch_sample_all()

    print(len(data_set.all_patients))

    print(data_set.max_visits)
    print(data_set.max_len_visit)

    sess = tf.compat.v1.Session(config=graph_config)
    with tf.compat.v1.variable_scope('mortality_prediction') as scope:
        if cfg.model_type == 'raw':
            model = RawModel(scope.name, data_set)
        elif cfg.model_type == 'tesa':
            ##############################################################################
            # TeSAN - proposed model
            ##############################################################################
            model = TesaModel(scope.name, data_set)
        elif cfg.model_type == 'delta':
            ##############################################################################
            # Interval - Ablation Studies
            ##############################################################################
            model = DeltaModel(scope.name, data_set)
        elif cfg.model_type == 'sa':
            ##############################################################################
            # Multi_Sa - Ablation Studies ??? by elimination a little less certain ???
            ##############################################################################
            model = SAModel(scope.name, data_set)
        elif cfg.model_type == 'normal':
            ##############################################################################
            # Normal_Sa - Ablation Studies
            ##############################################################################
            model = NormalModel(scope.name, data_set)
        elif cfg.model_type == 'cbow':
            ##############################################################################
            # CBOW - Baseline Method
            ##############################################################################
            model = CBOWModel(scope.name, data_set)
        elif cfg.model_type == 'sg':
            ##############################################################################
            # Skip-gram - Baseline Method
            ##############################################################################
            model = SGModel(scope.name, data_set)
        elif cfg.model_type == 'mce':
            ##############################################################################
            # MCE - Baseline Method  (CBOW variant)
            ##############################################################################
            model = MCEModel(scope.name, data_set)
        elif cfg.model_type == 'glove':
            ##############################################################################
            # GloVe - Baseline Method
            ##############################################################################
            model = GloveModel(scope.name, data_set)
        else:
            ##############################################################################
            # med2vec - Baseline Method
            ##############################################################################
            model = Med2VecModel(scope.name, data_set)

    graph_handler = GraphHandler(model, logging)
    graph_handler.initialize(sess)

    # evaluator = Evaluator(model,logging)
    global_steps = 0
    total_loss = 0
    logging.add()
    logging.add('Begin training...')

    for batch in sample_batches:

        feed_dict = {model.inputs: batch[0], model.labels: batch[1]}
        _, loss_val = sess.run([model.optimizer, model.loss], feed_dict=feed_dict)
        total_loss += loss_val
        global_steps += 1

        if global_steps % 10000 == 0:
            avg_loss = total_loss / 1000
            log_str = "Average loss at step %s: %s " % (global_steps, avg_loss)
            logging.add(log_str)
            total_loss = 0
            dev_feed_dict = {model.inputs: data_set.test_patients,
                             model.labels: data_set.test_labels}
            accuracy, props, yhat = sess.run([model.accuracy, model.props, model.yhat], feed_dict=dev_feed_dict)
            logging.add('validating the accuracy.....')
            log_str = "accuracy: %s" % accuracy
            logging.add(log_str)
            logging.add('validating more metrics.....')

            metrics = et.metric_pred(data_set.test_labels, props, yhat)
            log_str = "metrics: %s" % metrics
            logging.add(log_str)

            # save patient vectors
            # placehold = np.zeros((1, 100))
            # for batch in all_batches:
            #     feed_dict = {model.inputs: batch[0], model.labels: batch[1]}
            #     pat_embedding = sess.run(model.output, feed_dict=feed_dict)
            #     # print('pat_embedding shape:', pat_embedding.shape)
            #     placehold = np.append(placehold, pat_embedding, axis=0)
            #
            # placehold = np.delete(placehold, 0, 0)
            # print('placehold shape:', placehold.shape)
            # path = cfg.data_source + '_model_' + cfg.model +'_'+ str(global_steps) + '.patient.vect'
            # np.savetxt(join(cfg.saved_vect_dir, path), placehold, delimiter=',')
            # # placehold = np.zeros((1, 100))
            # # del placehold

    # # save patient vectors
    placehold = np.zeros((1, 100))
    for batch in all_batches:
        feed_dict = {model.inputs: batch[0], model.labels: batch[1]}
        pat_embedding = sess.run(model.output, feed_dict=feed_dict)
        # print('pat_embedding shape:', pat_embedding.shape)
        placehold = np.append(placehold, pat_embedding, axis=0)

    placehold = np.delete(placehold, 0, 0)
    print('placehold shape:', placehold.shape)
    path = cfg.data_source + '_model_' + cfg.model + '_'+str(num_steps)+'.patient.vect'
    np.savetxt(join(cfg.saved_vect_dir, path), placehold, delimiter=',')
    logging.done()


def test():
    pass


def main(_):
    if cfg.mode == 'train':
        train()
    elif cfg.mode == 'test':
        test()
    else:
        raise RuntimeError('no running mode named as %s' % cfg.mode)


def output_model_params():
    logging.add()
    logging.add('==>model_title: ' + cfg.model_name[1:])
    logging.add()
    for key,value in cfg.args.__dict__.items():
        if key not in ['test','shuffle']:
            logging.add('%s: %s' % (key, value))


if __name__ == '__main__':
    tf.compat.v1.app.run()



