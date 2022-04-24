import tensorflow as tf
import numpy as np
from os.path import join

from src.concept_embedding.configs import cfg

from src.concept_embedding.model.baseline.cbow import CBOWModel
from src.concept_embedding.model.ablation.delta import DeltaModel
from src.concept_embedding.model.fusion import FusionModel
from src.concept_embedding.model.ablation.normal import NormalModel
from src.concept_embedding.model.ablation.sa import SAModel
from src.concept_embedding.model.baseline.ta_attn import TaAttnModel
from src.concept_embedding.model.ablation.tesa import TesaNonDateModel
from src.concept_embedding.model.proposed.tesan import TeSANModel

from src.utils.graph_handler import GraphHandler
from src.utils.record_log import RecordLog
from src.concept_embedding.evaluation.evaluation import ConceptEvaluation as Evaluator
from src.concept_embedding.data.dataset import  ConceptDataset as CDataset
from src.concept_embedding.data.datasetEncodeDate import ConceptAndDateDataset as CDDataset

import warnings
warnings.filterwarnings('ignore')

logging = RecordLog()
logging.initialize(cfg)

def train():
    if cfg.globals["gpu_mem"] is None:
        gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=cfg.globals["gpu_mem"], allow_growth=True)
        graph_config = tf.compat.v1.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)
    else:
        gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=cfg.globals["gpu_mem"])
        graph_config = tf.compat.v1.ConfigProto(gpu_options=gpu_options)

    num_steps = cfg.data["num_steps"]

    if cfg.data["is_date_encoding"]:
        data_set = CDDataset()
        data_set.initialize(cfg.data)
        data_set.prepare_data()
        data_set.build_dictionary()
        data_set.build_dict4date()
        data_set.load_data()
        sample_batches = data_set.generate_batch(num_steps)
    else:
        data_set = CDataset()
        data_set.initialize(cfg.data)
        data_set.prepare_data()
        data_set.build_dictionary()
        data_set.load_data()
        sample_batches = data_set.generate_batch(num_steps)
        print('data_set.train_size =',data_set.train_size)
        batch_num = data_set.train_size / data_set.batch_size
        print('batch_num =',batch_num)

    sess = tf.compat.v1.Session(config=graph_config)
    with tf.compat.v1.variable_scope('concept_embedding') as scope:
        #model = Model(scope.name,data_set)
        if cfg.model == 'tesa': #TeSAN - the proposed
            if cfg.data["is_date_encoding"]:
                model = TeSANModel(scope.name, data_set)
            else:
                model = TesaNonDateModel(scope.name, data_set)

        elif cfg.model == 'delta':
            model = DeltaModel(scope.name, data_set)
        elif cfg.model == 'sa':
            model = SAModel(scope.name, data_set)
        elif cfg.model == 'normal':
            model = NormalModel(scope.name, data_set)

        elif cfg.model == 'cbow':
            model = CBOWModel(scope.name, data_set)
        elif cfg.model == 'ta_attn':
            model = TaAttnModel(scope.name, data_set)
        elif cfg.model == 'fusion':
            model = FusionModel(scope.name, data_set)


    graph_handler = GraphHandler(model,logging)
    graph_handler.initialize(sess, cfg)

    evaluator = Evaluator(model,logging)
    evaluator.initialize(cfg)

    global_step = 0
    total_loss = 0

    epoch_loss = 0
    tmp_epoch = 0
    tmp_cur_batch = 0

    logging.add()
    logging.add('Begin training...')
    for batch in sample_batches:
        if num_steps is not None: # run based on step number
            if cfg.data["is_date_encoding"]:
                feed_dict = {model.train_inputs: batch[0], model.train_labels: batch[1]}
            else:
                feed_dict = {model.train_inputs: batch[0], model.train_labels: batch[2], model.train_masks: batch[1]}
            _, loss_val = sess.run([model.optimizer, model.loss], feed_dict=feed_dict)
            total_loss += loss_val
            global_step += 1

            if global_step % 5000 == 0:
                avg_loss = total_loss / 1000
                print_eval("step", global_step, avg_loss, evaluator,sess)

                total_loss = 0
        else: # run based on epoch number
            if cfg.data["is_date_encoding"]:
                batch_num, current_epoch, current_batch = batch[2], batch[3], batch[4]
            else:
                batch_num, current_epoch, current_batch = batch[3], batch[4], batch[5]

            if tmp_epoch != current_epoch:
                epoch_loss /= tmp_cur_batch
                print_eval("epoch", tmp_epoch, epoch_loss, evaluator,sess)

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

            if cfg.data["is_date_encoding"]:
                feed_dict = {model.train_inputs: batch[0], model.train_labels: batch[1]}
            else:
                feed_dict = {model.train_inputs: batch[0], model.train_labels: batch[2], model.train_masks: batch[1]}

            _, loss_val = sess.run([model.optimizer, model.loss], feed_dict=feed_dict)
            epoch_loss += loss_val
    log_str = "\tTarget\t\t 32.84%\t 58.33%\t 66.10%\t 43.80%"
    logging.add(log_str)

    logging.done()

def print_eval(type, stage, loss, evaluator, sess):
    icd_weigh_scores = evaluator.get_clustering_nmi(sess, 'ICD')
    ccs_weigh_scores = evaluator.get_clustering_nmi(sess, 'CCS')

    icd_nns = evaluator.get_nns_p_at_top_k(sess, 'ICD')
    ccs_nns = evaluator.get_nns_p_at_top_k(sess, 'CCS')

    #logging.add('validating the embedding performance .....')
    if cfg.globals["verbose"]:
        if stage==0:
            header = "\t%s\tLoss\tNMI ICD\tNMI CCS\tNNS P@%s Score ICD\tNNS P@%s Score CCS" % (type, cfg.evaluation["top_k"], cfg.evaluation["top_k"])
            logging.add(header)
        log_str = "\t% 3d\t% 6.3f\t% 5.2f%%\t% 5.2f%%\t% 5.2f%%\t% 5.2f%%" % (stage, loss, icd_weigh_scores * 100, ccs_weigh_scores * 100, icd_nns * 100, ccs_nns * 100)
        logging.add(log_str)
    else:
        # logging.add("%s %s: % 6.3f%% (avg loss) " % (type, stage, loss))
        log_str = "weight: %s %s %s %s" % (icd_weigh_scores, ccs_weigh_scores, icd_nns, ccs_nns)
        logging.add(log_str)


def test():
    pass


def main(_):
    # logging.add('verbose = %s' % cfg.verbose)
    #
    # cfg.data["is_date_encoding"] = True
    # cfg.embedding_size = 100        #dimension d of the medical concept embedding was set to 100
    #
    # cfg.min_cut_freq = 5            # All infrequent medical concepts were removed and the threshold empirically set to 5
    # if cfg.data_source == 'mimic3':
    #     cfg.skip_window = 6         # The skip window of our model was empirically set to 6 for MIMIC III
    #     cfg.num_samples = 10        # number of negative samples in MIMIC III [...] was set to 10
    #     cfg.visit_threshold = 1
    #     cfg.train_batch_size = 64   # batch size is 64 for MIMIC III
    # else:
    #     cfg.skip_window = 7         # The skip window of our model was empirically set to [...] 7 for CMS
    #     cfg.num_samples = 5         # number of negative samples in [...] CMS was set to [...] 5
    #     cfg.visit_threshold = 4     # number of hospital visits was less than 4 in CMS were empirically discarded
    #     cfg.train_batch_size = 128  # batch size is [...] 128 for CMS

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

    train()

if __name__ == '__main__':
    tf.compat.v1.app.run()
    # --data_source mimic3 --model delta --gpu 2 --max_epoch 30 --num_steps 10000 --train_batch_size 64 --num_samples 10 --reduced_window True --skip_window 6 --verbose True --is_scale False --is_date_encoding False --task embedding
# --data_source mimic3 --model sa --gpu 1 --max_epoch 30 --train_batch_size 64 --num_samples 10 --reduced_window True --skip_window 6 --verbose True --is_scale False --is_date_encoding False --task embedding --visit_threshold 1
