import math
import collections
from numpy import sqrt
from os.path import join
import numpy as np
from sklearn.model_selection import train_test_split

from src.utils.file import save_file,load_file
import json


class MortalityDataset():

    def __init__(self, dataConfig, visit_threshold):
        self.word_sample = dict()
        self.reverse_dictionary = dict()
        self.dictionary = dict()
        self.words_count = 0
        self.total_visits = 0
        self.code_no_per_visit = 0
        self.max_len_visit = 0
        self.max_visits = 0
        self.train_size = 0

        self.patients_file = None
        self.dict_file = None
        self.days_size = None
        self.patients = None
        self.patients_codes_file = None

        self.train_patients = None
        self.train_labels = None
        self.all_patients = None
        self.all_labels = None
        self.test_patients = None
        self.test_labels = None
        self.train_size = 0

        self.dataset_dir = dataConfig["dataset_dir"]
        self.data_source = dataConfig["data_source"]
        self.is_sample = dataConfig["sample_flag"]
        self.sample_rate = dataConfig["sample_rate"]
        self.dx_only = dataConfig["only_dx_flag"]
        self.min_freq = dataConfig["min_cut_freq"]
        self.skip_window = dataConfig["skip_window"]
        self.batch_size = dataConfig["train_batch_size"]
        self.is_reduced_window = dataConfig["reduced_window"]
        self.dataset_dir = dataConfig["dataset_dir"]
        self.visit_threshold = visit_threshold

    def prepare_data(self):
            if self.data_source == 'mimic3':
                self.patients_file = join(self.dataset_dir, 'patients_mimic3_full.json')
                self.dict_file = join(self.dataset_dir, 'mimic3_dict')
                self.patients_codes_file = join(self.dataset_dir, 'patients_mimic3_codes')
                self.days_size = 12 * 365 + 1
            else:
                self.patients_file = join(self.dataset_dir, 'patients_cms_full.json')
                self.patients_codes_file = join(self.dataset_dir, 'patients_cms_codes')
                self.dict_file = join(self.dataset_dir, 'cms_dict')
                self.days_size = 4 * 365 + 1

            # logging.info('source data path:%s' % patients_file)
            with open(self.patients_file) as read_file:
                self.patients = json.load(read_file)

            # if not cfg.data_source == 'mimic3':
            #     self.patients = [patient for patient in self.patients if len(patient['visits']) >= visit_threshold]
            self.patients = [patient for patient in self.patients if len(patient['visits']) >= self.visit_threshold]

    def build_dictionary(self):

        all_codes = []  # store all diagnosis codes

        for patient in self.patients:
            for visit in patient['visits']:
                self.total_visits += 1
                dxs = visit['DXs']
                for dx in dxs:
                    all_codes.append('D_' + dx)
                if not self.dx_only:
                    txs = visit['CPTs']
                    for tx in txs:
                        all_codes.append('T_' + tx)

        # store all codes and corresponding counts
        count_org = []
        count_org.extend(collections.Counter(all_codes).most_common())

        # store filtering codes and counts
        count = []
        for word, c in count_org:
            word_tuple = [word, c]
            if c >= self.min_freq:
                count.append(word_tuple)
                self.words_count += c

        if not self.sample_rate:
            # no words downsampled
            threshold_count = self.words_count
        elif self.sample_rate < 1.0:
            # traditional meaning: set parameter as proportion of total
            threshold_count = self.sample_rate * self.words_count
        else:
            # new shorthand: sample >= 1 means downsample all words with higher count than sample
            threshold_count = int(self.sample_rate * (3 + sqrt(5)) / 2)

        self.code_no_per_visit = self.words_count / self.total_visits
        # add padding
        self.dictionary['PAD'] = 0
        for word, cnt in count:
            index = len(self.dictionary)
            self.dictionary[word] = index
            word_probability = (sqrt(cnt / threshold_count) + 1) * (threshold_count / cnt)
            sample_int = int(round(word_probability * 2 ** 32))
            self.word_sample[index] = int(sample_int)

        # encoding patient using index
        for patient in self.patients:
            visits = patient['visits']
            len_visits = len(visits)
            if len_visits > self.max_visits:
                self.max_visits = len_visits
            for visit in visits:
                dxs = visit['DXs']
                if len(dxs) == 0:
                    continue
                else:
                    visit['DXs'] = [self.dictionary['D_' + dx] for dx in dxs if 'D_' + dx in self.dictionary]

                if not self.dx_only:
                    txs = visit['CPTs']
                    if len(txs) == 0:
                        continue
                    else:
                        visit['CPTs'] = [self.dictionary['T_' + tx] for tx in txs if 'T_' + tx in self.dictionary]
                len_current_visit = len(visit['DXs'])
                if len_current_visit > self.max_len_visit:
                    self.max_len_visit = len_current_visit

        self.reverse_dictionary = dict(zip(self.dictionary.values(), self.dictionary.keys()))

        with open(self.dict_file + '.json', 'w') as fp:
            json.dump(self.reverse_dictionary, fp)

        with open(self.patients_codes_file + '.json', 'w') as fp:
            json.dump(self.patients, fp)

    def process_data(self):

        batches = []
        patient_dict = {}
        index = 0
        for patient in self.patients:
            # get patient's visits
            patient_dict['pid_'+patient['pid']] = index
            index += 1
            visits = patient['visits']
            # sorting visits by admission date
            sorted_visits = sorted(visits, key=lambda visit: visit['admsn_dt'])

            # number of visits
            no_visits = len(visits)

            # generating batch sample: list of visits including concept_embedding codes,
            # label of last visit mortality
            ls_visits = []
            label = [int(sorted_visits[no_visits-1]['Death'])]
            for visit in sorted_visits:
                codes = visit['DXs']
                if not self.dx_only:
                    codes.extend(visit['CPTs'])

                code_size = len(codes)
                # code padding
                if code_size < self.max_len_visit:
                    list_zeros = [0] * (self.max_len_visit - code_size)
                    codes.extend(list_zeros)
                ls_visits.append(codes)

            # visit padding
            if no_visits < self.max_visits:
                for i in range(self.max_visits - no_visits):
                    list_zeros = [0] * self.max_len_visit
                    ls_visits.append(list_zeros)
            # print(len(ls_visits))
            batches.append([np.array(ls_visits, dtype=np.int32), np.array(label, dtype=np.int32)])

        b_patients = []
        b_label = []
        for batch in batches:
            b_patients.append(batch[0])
            b_label.append(batch[1])

        save_file({'patient': b_patients, 'label': b_label},
                  join(self.dataset_dir, 'mortality_'+self.data_source+'.pickle'))

        dict_file = join(self.dataset_dir, 'mimic3_patient_dict')
        print('patient dict file location: ', dict_file)
        with open(dict_file + '.json', 'w') as fp:
            json.dump(patient_dict, fp)

        return b_patients, b_label

    def load_data(self):

        processed_file = join(self.dataset_dir, 'mortality_'+self.data_source+'.pickle')
        is_load, data = load_file(processed_file, 'processed data', 'pickle')

        if not is_load:
            data = self.process_data()
            patients = data[0]
            labels = data[1]
            self.all_patients = patients
            self.all_labels = labels
        else:
            patients = data['patient']
            labels = data['label']
            self.all_patients = patients
            self.all_labels = labels

        self.train_patients, self.test_patients, self.train_labels, self.test_labels\
            = train_test_split(patients, labels, test_size=0.1, random_state=42)

        self.train_size = len(self.train_patients)
        #
        # self.dev_patients, self.test_patients, self.dev_labels, self.test_labels \
        #     = train_test_split(vt_patients, vt_labels, test_size=0.5, random_state=42)

    def generate_batch(self, num_steps):

        def data_queue(train_patients, train_labels, batch_size):
            assert len(train_patients) >= batch_size
            data_ptr = 0
            data_round = 0
            idx_b = 0
            step = 0
            while True:
                if data_ptr + batch_size <= len(train_patients):
                    batch_patients = train_patients[data_ptr:data_ptr + batch_size]
                    batch_labels = train_labels[data_ptr:data_ptr + batch_size]

                    yield np.array(batch_patients, dtype=np.int32), \
                          np.array(batch_labels, dtype=np.int32), \
                          data_round, idx_b
                    data_ptr += batch_size
                    idx_b += 1
                    step += 1
                elif data_ptr + batch_size > len(train_patients):
                    offset = data_ptr + batch_size - len(train_patients)

                    batch_patients = train_patients[data_ptr:]
                    batch_patients += train_patients[:offset]
                    batch_labels = train_labels[data_ptr:]
                    batch_labels += train_labels[:offset]

                    data_ptr = offset
                    data_round += 1

                    yield np.array(batch_patients, dtype=np.int32),\
                          np.array(batch_labels,dtype=np.int32), data_round, 0
                    idx_b = 1
                    step += 1
                if step >= num_steps:
                    break

        batch_num = math.ceil(self.train_size / self.batch_size)
        for patients, labels, data_round, idx_b in data_queue(self.train_patients,
                                                              self.train_labels,
                                                              self.batch_size):
            yield patients, labels, batch_num, data_round, idx_b

    # def generate_batch_sample_iter(self):
    #
    #     batch_num = math.ceil(len(self.test_patients) / self.batch_size)
    #
    #     def data_queue():
    #         assert len(self.test_patients) >= self.batch_size
    #         data_ptr = 0
    #         data_round = 0
    #         idx_b = 0
    #         step = 0
    #         while True:
    #             if data_ptr + self.batch_size <= len(self.test_patients):
    #                 batch_patients = self.test_patients[data_ptr:data_ptr + self.batch_size]
    #                 batch_labels = self.test_labels[data_ptr:data_ptr + self.batch_size]
    #
    #                 yield np.array(batch_patients, dtype=np.int32), \
    #                       np.array(batch_labels, dtype=np.int32), \
    #                       data_round, idx_b
    #                 data_ptr += self.batch_size
    #                 idx_b += 1
    #                 step += 1
    #             elif data_ptr + self.batch_size > len(self.test_patients):
    #                 offset = data_ptr + self.batch_size - len(self.test_patients)
    #
    #                 batch_patients = self.test_patients[data_ptr:]
    #                 batch_patients += self.test_patients[:offset]
    #                 batch_labels = self.test_labels[data_ptr:]
    #                 batch_labels += self.test_labels[:offset]
    #
    #                 data_ptr = offset
    #                 data_round += 1
    #
    #                 yield np.array(batch_patients, dtype=np.int32),\
    #                       np.array(batch_labels,dtype=np.int32), data_round, 0
    #                 idx_b = 1
    #                 step += 1
    #             if step >= batch_num:
    #                 break
    #
    #     for patients, labels, data_round, idx_b in data_queue():
    #         yield patients, labels, batch_num, data_round, idx_b

    def generate_batch_sample_all(self):

        batch_num = math.ceil(len(self.all_patients) / self.batch_size)

        def data_queue():
            assert len(self.all_patients) >= self.batch_size
            data_ptr = 0
            data_round = 0
            idx_b = 0
            step = 0
            while True:
                if data_ptr + self.batch_size <= len(self.all_patients):
                    batch_patients = self.all_patients[data_ptr:data_ptr + self.batch_size]
                    batch_labels = self.all_labels[data_ptr:data_ptr + self.batch_size]

                    yield np.array(batch_patients, dtype=np.int32), \
                          np.array(batch_labels, dtype=np.int32), \
                          data_round, idx_b, step
                    data_ptr += self.batch_size
                    idx_b += 1
                    step += 1
                elif data_ptr + self.batch_size > len(self.all_patients):
                    offset = data_ptr + self.batch_size - len(self.all_patients)

                    batch_patients = self.all_patients[data_ptr:]
                    # batch_patients += self.all_patients[:offset]
                    batch_labels = self.all_labels[data_ptr:]
                    # batch_labels += self.all_labels[:offset]

                    data_ptr = offset
                    data_round += 1

                    yield np.array(batch_patients, dtype=np.int32),\
                          np.array(batch_labels,dtype=np.int32), data_round, 0, step
                    idx_b = 1
                    step += 1
                if step >= batch_num:
                    break

        for patients, labels, data_round, idx_b, step in data_queue():
            yield patients, labels, step, batch_num, data_round, idx_b