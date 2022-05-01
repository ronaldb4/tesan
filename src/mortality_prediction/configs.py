import argparse
import json
import os
import warnings

from os.path import join

#
# class to set, extract and encapsulate run-time parameters for the models
# to do so, it parses the command-line parameters
# additionally, it will created directories as needed
#
class Configs(object):
    def __init__(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # ------ parse input arguments--------
        parser = argparse.ArgumentParser()
        parser.register('type', 'bool', (lambda x: x.lower() in ('True', "yes", "true", "t", "1")))

        parser.add_argument('--model', type=str, default='tesa', help='tesan, cbow, normal, sa, delta, tesa')
        parser.add_argument('--configSet', type=str, default='mimic_paper.json', help='.json file containing configSet')

        parser.set_defaults(shuffle=True)
        self.args = parser.parse_args()

        self.model = self.args.model
        self.configSet = self.args.configSet

        # ------ set run-time parameters --------
        with open(''.join(['../',self.configSet])) as d:
            configSet = json.load(d)

        self.project_dir = self.getRootDir()

        self.globals  = configSet['globals']

        self.data = configSet['data']
        self.data["dataset_dir"]     = join(self.project_dir, 'dataset', 'processed')

        self.evaluation = configSet['evaluation']

        self.modelParams = configSet['models'][self.model]

        #------------------path-------------------------------
        self.standby_log_dir = self.mkdir(self.project_dir, 'logs')
        self.result_dir      = self.mkdir(self.project_dir, 'outputs/mortality_prediction')

        self.model_dir          = self.mkdir(self.result_dir, self.model)
        self.summary_dir        = self.mkdir(self.model_dir, 'summary')
        self.ckpt_dir           = self.mkdir(self.model_dir, 'ckpt')
        self.log_dir            = self.mkdir(self.model_dir, 'log_files')
        self.saved_vect_dir     = self.mkdir(self.model_dir, 'vects')
        self.processed_dir      = self.mkdir(self.model_dir, 'processed_data')

        self.dict_dir           = self.mkdir(self.result_dir, 'dict')

        processed_name = '_'.join([self.model, self.data["data_source"], "sw", str(self.data["skip_window"])]) + '.pickle'
        self.data["processed_path"] = join(self.processed_dir, processed_name)

        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.globals["gpu"])

        self.log_name = 'ds_' + self.data["data_source"] + \
                        "_m_" + self.model + \
                        "_ns_" + str(self.evaluation["num_steps"])


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
