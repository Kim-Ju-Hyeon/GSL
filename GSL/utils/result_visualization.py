from glob import glob, escape
import pickle
import yaml
from easydict import EasyDict as edict


def get_exp_result_files(exp):
    config_file = glob(escape(exp + '/config.yaml'))[0]
    config = edict(yaml.load(open(config_file, 'r'), Loader=yaml.FullLoader))

    train_result_dirs = glob(escape(exp + '/training_result.pickle'))
    train_result = pickle.load(open(train_result_dirs[0], 'rb'))

    test_result_dirs = glob(escape(exp + '/test_result.pickle'))
    test_result = pickle.load(open(test_result_dirs[0], 'rb'))

    return config, train_result, test_result
