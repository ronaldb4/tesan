import datetime
import os
import time
import tensorflow as tf
import numpy as np
from os.path import join
import psutil

from src.concept_embedding.configs import cfg

from src.concept_embedding.model.baseline.cbow import CBOWModel
from src.concept_embedding.model.baseline.skip_gram import SkipGramModel
from src.concept_embedding.model.ablation.delta import DeltaModel
from src.concept_embedding.model.fusion import FusionModel
from src.concept_embedding.model.ablation.normal import NormalModel
from src.concept_embedding.model.ablation.sa import SAModel
from src.concept_embedding.model.ablation.random_interval import RandomIntervalModel
from src.concept_embedding.model.baseline.ta_attn import TaAttnModel
from src.concept_embedding.model.ablation.tesa import TesaNonDateModel
from src.concept_embedding.model.proposed.tesan import TeSANModel

from src.utils.graph_handler import GraphHandler
from src.utils.record_log import RecordLog
from src.concept_embedding.evaluation.evaluation import ConceptEvaluation as Evaluator
from src.concept_embedding.data.datasetRandomInterval import  ConceptRandomIntervalDataset as CRandIntDataset
from src.concept_embedding.data.datasetSkipGram import  ConceptSkipDataset as CSkipDataset
from src.concept_embedding.data.dataset import  ConceptDataset as CDataset
from src.concept_embedding.data.datasetEncodeDate import ConceptAndDateDataset as CDDataset

import warnings
warnings.filterwarnings('ignore')

logging = RecordLog()
logging.initialize(cfg)

def train():
    process = psutil.Process(os.getpid())

    num_steps = cfg.data["num_steps"]

    if cfg.model == 'tesan':
        data_set = CDDataset()
        data_set.initialize(cfg.data)
        data_set.prepare_data()
        data_set.build_dictionary()
        data_set.build_dict4date()
        data_set.load_data()
    else:
        if cfg.model == 'skip_gram':
            data_set = CSkipDataset()
        elif cfg.model == 'random_interval':
            data_set = CRandIntDataset()
        else:
            data_set = CDataset()
        data_set.initialize(cfg.data)
        data_set.prepare_data()
        data_set.build_dictionary()
        data_set.load_data()

    if cfg.globals["gpu_mem"] is None:
        gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=cfg.globals["gpu_mem"], allow_growth=True)
        graph_config = tf.compat.v1.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)
    else:
        gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=cfg.globals["gpu_mem"])
        graph_config = tf.compat.v1.ConfigProto(gpu_options=gpu_options)

    sess = tf.compat.v1.Session(config=graph_config)

    with tf.compat.v1.variable_scope('concept_embedding') as scope:
        if cfg.model == 'tesan': #TeSAN - the proposed
            model = TeSANModel(scope.name, data_set)

        #ablation models
        elif cfg.model == 'tesa': #Multi_SA???? performance is inline
            model = TesaNonDateModel(scope.name, data_set)
        elif cfg.model == 'delta': #Interval
            model = DeltaModel(scope.name, data_set)
        elif cfg.model == 'sa': #Multi_SA or Normal_SA
            model = SAModel(scope.name, data_set)
        elif cfg.model == 'normal': #Normal_SA or something else
            model = NormalModel(scope.name, data_set)
        elif cfg.model == 'random_interval':  # supplementary
            model = RandomIntervalModel(scope.name, data_set)

        #baseline models
        elif cfg.model == 'cbow':
            model = CBOWModel(scope.name, data_set)
        elif cfg.model == 'skip_gram':
            model = SkipGramModel(scope.name, data_set)

        # ????
        elif cfg.model == 'ta_attn':
            model = TaAttnModel(scope.name, data_set)
        elif cfg.model == 'fusion':
            model = FusionModel(scope.name, data_set)

    graph_handler = GraphHandler(model,logging)
    graph_handler.initialize(sess, cfg)

    evaluator = Evaluator(model,logging)
    evaluator.initialize(cfg)

    epoch_loss = 0
    tmp_epoch = 0
    tmp_cur_batch = 0

    logging.add()
    logging.add('Begin training...')
    sample_batches = data_set.generate_batch(num_steps)

    total_cpu = 0
    memory_usage = []
    if cfg.globals["verbose"]:
        header = "\t%s\tLoss\tNMI ICD\tNMI CCS\tNNS P@%s Score ICD\tNNS P@%s Score CCS\tDuration\tCPU (sec)\tMemory (MB)" % (
        "epoch", cfg.evaluation["top_k"], cfg.evaluation["top_k"])
        logging.add(header)

    for batch in sample_batches:
        if cfg.model == 'tesan':
            batch_num, current_epoch, current_batch = batch[2], batch[3], batch[4]
        else:
            batch_num, current_epoch, current_batch = batch[3], batch[4], batch[5]

        if tmp_epoch != current_epoch:
            epoch_end = time.perf_counter()
            epoch_loss /= tmp_cur_batch
            cpu_time, memory_used = print_eval(tmp_epoch, epoch_loss, evaluator,sess, process)
            total_cpu = cpu_time
            memory_usage.append(memory_used)
            # print('avg mem: % 7.2f (MB)' % ((sum(memory_usage) / 1024 / 1024) / len(memory_usage)))
            # print('max mem: % 7.2f (MB)' % ((max(memory_usage) / 1024 / 1024)))
            epoch_loss = 0
            tmp_epoch = current_epoch
        else:
            tmp_cur_batch = current_batch

        if current_epoch == cfg.evaluation["max_epoch"]:
            embeddings = sess.run(model.final_weights)

            path = cfg.data["data_source"] + '_model_' + cfg.model + '_epoch_' + \
                   str(cfg.evaluation["max_epoch"]) + '_sk_' + str(cfg.data["skip_window"]) + '.vect'
            np.savetxt(join(cfg.saved_vect_dir, path), embeddings, delimiter=',')
            break

        if cfg.model == 'tesan':
            feed_dict = {model.train_inputs: batch[0], model.train_labels: batch[1]}
        else:
            feed_dict = {model.train_inputs: batch[0], model.train_labels: batch[2], model.train_masks: batch[1]}

        _, loss_val = sess.run([model.optimizer, model.loss], feed_dict=feed_dict)
        epoch_loss += loss_val

    logging.add('total cpu time: %s' % str(datetime.timedelta(seconds=total_cpu)))
    logging.add('avg mem: % 7.2f (MB)' % ((sum(memory_usage)/1024/1024)/len(memory_usage)))
    logging.add('max mem: % 7.2f (MB)' % ((max(memory_usage)/1024/1024)))

    logging.done()

def print_eval(stage, loss, evaluator, sess, process):
    icd_weigh_scores = evaluator.get_clustering_nmi(sess, 'ICD')
    ccs_weigh_scores = evaluator.get_clustering_nmi(sess, 'CCS')

    icd_nns = evaluator.get_nns_p_at_top_k(sess, 'ICD')
    ccs_nns = evaluator.get_nns_p_at_top_k(sess, 'CCS')

    cpu_time = process.cpu_times().user + psutil.cpu_times().system
    memory_used = process.memory_info().vms
    if cfg.globals["verbose"]:
        log_str = "\t% 3d\t% 6.3f\t% 5.2f%%\t% 5.2f%%\t% 5.2f%%\t% 5.2f%%\t% 7.2f\t% 7.2f" % \
                  (stage, loss, icd_weigh_scores * 100, ccs_weigh_scores * 100, icd_nns * 100, ccs_nns * 100, cpu_time, memory_used/1024/1024)
        logging.add(log_str)
    else:
        log_str = "weight: %s %s %s %s %s %s" % (icd_weigh_scores, ccs_weigh_scores, icd_nns, ccs_nns, cpu_time, memory_used)
        logging.add(log_str)

    return cpu_time, memory_used

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

def main(_):
    train()


if __name__ == '__main__':
    tf.compat.v1.app.run()
