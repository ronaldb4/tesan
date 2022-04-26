import argparse
import json
import os
import warnings

from os.path import join


class Configs(object):
    def __init__(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # ------parsing input arguments"--------
        parser = argparse.ArgumentParser()
        parser.register('type', 'bool', (lambda x: x.lower() in ('True', "yes", "true", "t", "1")))

        parser.add_argument('--model', type=str, default='tesa', help='tesa, vanila_sa, or cbow')
        parser.add_argument('--configSet', type=str, default='mimic_paper.json', help='tesa, vanila_sa, or cbow')

        parser.set_defaults(shuffle=True)
        self.args = parser.parse_args()

        self.model = self.args.model
        self.configSet = self.args.configSet

        with open(self.configSet) as d:
            configSet = json.load(d)

        self.project_dir = self.getRootDir()

        self.globals  = configSet['globals']

        self.data = configSet['data']
        self.data["dataset_dir"]     = join(self.project_dir, 'dataset', 'processed')

        self.evaluation = configSet['evaluation']
        self.evaluation["icd_file"] = join(self.project_dir, 'src/utils/ontologies/D_ICD_DIAGNOSES.csv')
        self.evaluation["ccs_file"] = join(self.project_dir, 'src/utils/ontologies/SingleDX-edit.txt')
        self.evaluation["valid_examples"] = list(range(1, self.evaluation["valid_size"] + 1))  # Only pick dev samples in the head of the distribution.

        self.modelParams = configSet['models'][self.model]

        #------------------path-------------------------------
        self.standby_log_dir = self.mkdir(self.project_dir, 'logs')
        self.result_dir      = self.mkdir(self.project_dir, 'outputs/concept_embedding')
        self.all_model_dir   = self.mkdir(self.result_dir, 'tasks')

        self.model_dir          = self.mkdir(self.all_model_dir, self.globals["task"], self.model)
        self.summary_dir        = self.mkdir(self.model_dir, 'summary')
        self.ckpt_dir           = self.mkdir(self.model_dir, 'ckpt')
        self.log_dir            = self.mkdir(self.model_dir, 'log_files')
        self.saved_vect_dir     = self.mkdir(self.model_dir, 'vects')

        self.dict_dir           = self.mkdir(self.result_dir, 'dict')
        self.processed_dir      = self.mkdir(self.result_dir, 'processed_data')

        self.processed_task_dir = self.mkdir(self.processed_dir, self.globals["task"])

        # self.processed_name = '_'.join([self.model, self.data_source, str(self.skip_window), self.task, self.task_type]) + '.pickle'
        processed_name = '_'.join([self.model, self.data["data_source"], str(self.data["skip_window"]), self.globals["task"]]) + '.pickle'
        if self.data["is_date_encoding"]:
            processed_name = '_'.join([self.model, self.data["data_source"],str(self.data["skip_window"]),'withDateEncoding'])+'.pickle'
        print(processed_name)
        self.data["processed_path"] = join(self.processed_task_dir, processed_name)

        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.globals["gpu"])

        self.log_name = 'ds_' + self.data["data_source"] + \
                        "_m_" + self.model + \
                        "_me_" + str(self.evaluation["max_epoch"]) + \
                        "_tbs_" + str(self.data["train_batch_size"])


    def mkdir(self, *args):
        dir_path = join(*args)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        return dir_path

    def get_file_name_from_path(self, path):
        assert isinstance(path, str)
        file_name = '.'.join((path.split('/')[-1]).split('.')[:-1])
        return file_name

    def getRootDir(self):
        root_dir, _ = os.path.split(os.path.abspath(__file__))
        root_dir = os.path.dirname(root_dir)
        root_dir = os.path.dirname(root_dir)
        # root_dir = os.path.dirname(root_dir)
        return root_dir


cfg = Configs()
