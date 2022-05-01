import datetime
import os
import time
import tensorflow as tf
import numpy as np
from os.path import join
import psutil

from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score, auc

from src.mortality_prediction.model.baseline.cbow import CBOWModel
from src.mortality_prediction.model.baseline.glove import GloveModel
from src.mortality_prediction.model.baseline.mce import MCEModel
from src.mortality_prediction.model.baseline.med2vec import Med2VecModel
from src.mortality_prediction.model.baseline.sg import SGModel
from src.mortality_prediction.model.baseline.raw import RawModel
from src.mortality_prediction.model.ablation.tesa import TesaModel
from src.mortality_prediction.model.proposed.tesan import TeSANModel
from src.mortality_prediction.model.ablation.delta import DeltaModel
from src.mortality_prediction.model.ablation.sa import SAModel
from src.mortality_prediction.model.ablation.normal import NormalModel
from src.mortality_prediction.configs import cfg

from src.utils.graph_handler import GraphHandler
from src.utils.record_log import RecordLog
from src.mortality_prediction.data.dataset import MortalityDataset

import warnings
warnings.filterwarnings('ignore')
logging = RecordLog()
logging.initialize(cfg)


def train():
    process = psutil.Process(os.getpid())

    if cfg.globals["gpu_mem"] is None:
        gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=cfg.globals["gpu_mem"], allow_growth=True)
        graph_config = tf.compat.v1.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)

    else:
        gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=cfg.globals["gpu_mem"])
        graph_config = tf.compat.v1.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)

                            ########################################################
    visit_threshold = 1     # this is mildly interesting - or not
                            ########################################################
    num_steps = cfg.evaluation["num_steps"]
    data_set = MortalityDataset(cfg.data, visit_threshold)
    data_set.prepare_data()
    data_set.build_dictionary()
    data_set.load_data()
    sample_batches = data_set.generate_batch(num_steps)

    all_batches = data_set.generate_batch_sample_all()

    print(len(data_set.all_patients))

    print(data_set.max_visits)
    print(data_set.max_len_visit)

    sess = tf.compat.v1.Session(config=graph_config)
    with tf.compat.v1.variable_scope('mortality_prediction') as scope:
        if cfg.model == 'raw':
            model = RawModel(scope.name, data_set, cfg.modelParams)
        elif cfg.model == 'tesan':
            ##############################################################################
            # TeSAN - proposed model
            ##############################################################################
            model = TeSANModel(scope.name, data_set, cfg.modelParams)
        elif cfg.model == 'tesa':
            ##############################################################################
            # TeSAN - proposed model
            ##############################################################################
            model = TesaModel(scope.name, data_set, cfg.modelParams)
        elif cfg.model == 'delta':
            ##############################################################################
            # Interval - Ablation Studies
            ##############################################################################
            model = DeltaModel(scope.name, data_set, cfg.modelParams)
        elif cfg.model == 'sa':
            ##############################################################################
            # Multi_Sa - Ablation Studies ??? by elimination a little less certain ???
            ##############################################################################
            model = SAModel(scope.name, data_set, cfg.modelParams)
        elif cfg.model == 'normal':
            ##############################################################################
            # Normal_Sa - Ablation Studies
            ##############################################################################
            model = NormalModel(scope.name, data_set, cfg.modelParams)
        elif cfg.model == 'cbow':
            ##############################################################################
            # CBOW - Baseline Method
            ##############################################################################
            model = CBOWModel(scope.name, data_set, cfg.modelParams)
        elif cfg.model == 'sg':
            ##############################################################################
            # Skip-gram - Baseline Method
            ##############################################################################
            model = SGModel(scope.name, data_set, cfg.modelParams)
        elif cfg.model == 'mce':
            ##############################################################################
            # MCE - Baseline Method  (CBOW variant)
            ##############################################################################
            model = MCEModel(scope.name, data_set, cfg.modelParams)
        elif cfg.model == 'glove':
            ##############################################################################
            # GloVe - Baseline Method
            ##############################################################################
            model = GloveModel(scope.name, data_set, cfg.modelParams)
        else:
            ##############################################################################
            # med2vec - Baseline Method
            ##############################################################################
            model = Med2VecModel(scope.name, data_set, cfg.modelParams)

    graph_handler = GraphHandler(model, logging)
    graph_handler.initialize(sess,cfg)

    # evaluator = Evaluator(model,logging)
    global_steps = 0
    total_loss = 0
    logging.add()
    logging.add('Begin predicting...')

    if cfg.globals["verbose"]:
        header = "\tStep\tAccuracy\tPrecision\tSensitivity\tSpecificity\tF-score\tPR_AUC\tF1\tCPU\tMemory"
        logging.add(header)

    total_cpu = 0
    memory_usage = []
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

            cpu_time = process.cpu_times().user + psutil.cpu_times().system
            memory_used = process.memory_info().vms
            total_cpu = cpu_time
            memory_usage.append(memory_used)

            metrics = metric_pred(data_set.test_labels, props, yhat)

            if cfg.globals["verbose"]:
                #accuracy, precision, sensitivity, specificity, f_score, pr_auc, f1
                log_str = "% 6d\t% 3.2f%%\t% 3.2f%%\t% 3.2f%%\t% 3.2f%%\t% 3.2f%%\t% 3.2f%%\t% 3.2f%%\t% 7.2f\t% 7.2f" % \
                          (global_steps, metrics[0]*100, metrics[1], metrics[2], metrics[3], metrics[4], metrics[5], metrics[6],
                           cpu_time, memory_used / 1024 / 1024)
                logging.add(log_str)
            else:
                log_str = "metrics: %s %s %s" % (metrics, cpu_time, memory_used)
                logging.add(log_str)

    logging.add('total cpu time: %s' % str(datetime.timedelta(seconds=total_cpu)))
    logging.add('avg mem: % 7.2f' % sum(memory_usage/1024/1024)/len(memory_usage))
    logging.add('max mem: % 7.2f' % max(memory_usage/1024/1024))

    # # save patient vectors
    placehold = np.zeros((1, 100))
    for batch in all_batches:
        feed_dict = {model.inputs: batch[0], model.labels: batch[1]}
        pat_embedding = sess.run(model.output, feed_dict=feed_dict)
        # print('pat_embedding shape:', pat_embedding.shape)
        placehold = np.append(placehold, pat_embedding, axis=0)

    placehold = np.delete(placehold, 0, 0)
    print('placehold shape:', placehold.shape)
    path = cfg.data["data_source"] + '_model_' + cfg.model + '_'+str(num_steps)+'.patient.vect'
    np.savetxt(join(cfg.saved_vect_dir, path), placehold, delimiter=',')
    logging.done()


def log_config():
    logging.add()
    logging.add('execution config')
    for key,value in cfg.globals.items():
        logging.add('\t%s: %s' % (key, value))

    logging.add()
    logging.add('data config')
    for key,value in cfg.data.items():
        logging.add('\t%s: %s' % (key, value))

    logging.add()
    logging.add('evaluation config')
    for key,value in cfg.evaluation.items():
        logging.add('\t%s: %s' % (key, value))

    logging.add(cfg.model, ' config')
    for key,value in cfg.modelParams.items():
        logging.add('\t%s: %s' % (key, value))


def metric_pred(y_true, probs, y_pred):
    [[TN, FP], [FN, TP]] = confusion_matrix(y_true, y_pred, labels=[0, 1]).astype(float)
    # print(TN, FP, FN, TP)
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    specificity = TN / (FP + TN)
    precision = TP / (TP + FP)
    sensitivity = recall = TP / (TP + FN)
    f_score = 2 * TP / (2 * TP + FP + FN)

    # calculate AUC
    # roc_auc = roc_auc_score(y_true, probs)
    # print('roc_auc: %.4f' % roc_auc)
    # calculate roc curve
    # fpr, tpr, thresholds = roc_curve(y_true, probs)

    # calculate precision-recall curve
    precision_curve, recall_curve, thresholds = precision_recall_curve(y_true, probs)

    # calculate F1 score
    f1 = f1_score(y_true, y_pred)
    # calculate precision-recall AUC
    pr_auc = auc(recall_curve, precision_curve)

    return [accuracy, precision, sensitivity, specificity, f_score, pr_auc, f1]

    # return [accuracy, precision, sensitivity, specificity, f_score, roc_auc, pr_auc, f1]


def main(_):
    train()

if __name__ == '__main__':
    tf.compat.v1.app.run()
