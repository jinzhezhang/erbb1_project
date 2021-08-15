import sys
import timeit
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import roc_auc_score
import pickle

# import models
# import preprocess as pp
# import config
# from models import MolecularGraphNeuralNetwork, Trainer, Tester
# import sys
# sys.path.append('/home/zhang/elix')

from . import models
from . import preprocess as pp

def load_gnn_model(path, device):
    model = torch.load(path)
    model.to(device)
    return model

def load_gnn_dicts(path):
    atom_file = path + 'atom_dict.pkl'
    bond_file = path + 'bond_dict.pkl'
    fp_file = path + 'fp_dict.pkl'
    edge_file = path + 'edge_dict.pkl'
    if os.path.isfile(atom_file):
        print('load atom')
        atom_dict = pickle.load(open(atom_file,'rb'))
    if os.path.isfile(bond_file):
        print('load bond')
        bond_dict = pickle.load(open(bond_file, 'rb'))
    if os.path.isfile(fp_file):
        print('load fp')
        fingerprint_dict = pickle.load(open(fp_file, 'rb'))
    if os.path.isfile(edge_file):
        print('load edge')
        edge_dict = pickle.load(open(fp_file, 'rb'))
    #print(atom_dict)
    dicts = [atom_dict, bond_dict, fingerprint_dict, edge_dict]
    return dicts

def predict(model, task, path, radius, dataset, device, smiles_list):
    import gnn.main.preprocess as pp
    with open( path + 'testing_batch.txt', 'w') as f:
        for smiles in smiles_list:
            f.write(smiles + ' 0.\n')

    dataset_test = pp.create_datasets(task, dataset, radius, device,
                                    specific_file = path + 'testing_batch.txt')
    res = model.forward_regressor(list(zip(*dataset_test)), train = False)
    return res[0]

def predict_vec(model, task, path, radius, dataset, device, smiles_list, dicts):
    vec = pp.preprocess_single_mol(task, smiles_list,
                                    torch.Tensor([0. ]), radius, device, dicts)
    res = model.forward_regressor(list(zip(*vec)), train = False)
    print(smiles_list[0], '|', res[0][0])
    return res[0]

if __name__ == "__main__":

    if len(sys.argv) > 1:
        # bash execute
        (config.task, config.dataset, config.radius, config.dim,
        config.layer_hidden, config.layer_output, config.batch_train,
        config.batch_test, config.lr, config.lr_decay,
        config.decay_interval, config.iteration, setting) = sys.argv[1:]
        (config.radius, config.dim, config.layer_hidden, config.layer_output,
         config.batch_train, config.batch_test, config.decay_interval,
         config.iteration) = map(int, [config.radius, config.dim,
                                config.layer_hidden, config.layer_output,
                                config.batch_train, config.batch_test,
                                config.decay_interval, config.iteration])

        config.lr, config.lr_decay = map(float, [config.lr, config.lr_decay])

    torch.manual_seed(1234)
    np.random.seed(1234)

    model = torch.load("../model/200_gnn.pth")
    model.to(config.device)
    trainer = Trainer(model)
    tester = Tester(model)
    dicts = load_gnn_dicts()

    print(
        predict_vec(
            model = model,
            task = 'regression',
            path = '../dataset/regression/erbb1_clean_log_ic50/',
            radius = config.radius,
            dataset = 'erbb1_clean_log_ic50',
            device = config.device,
            smiles_list = ["C=C(CN1CCOCC1)C(=O)N1CC(Oc2cc3c(Nc4ccc(F)c(Cl)c4F)ncnc3cc2OC)C1"],
            dicts = dicts
        )
    )
    # if config.task == 'classification':
    #     result = 'Epoch\tTime(sec)\tLoss_train\tAUC_dev\tAUC_test'
    # if config.task == 'regression':
    #     result = 'Epoch\tTime(sec)\tLoss_train\tMAE_dev\tMAE_test'

    # dataset_test = pp.create_datasets(task, dataset, radius, device, specific_file = 'testing_batch.txt')
    # if task == 'classification':
    #     prediction_dev = tester.test_classifier(dataset_dev)
    #     prediction_test = tester.test_classifier(dataset_test)
    # if task == 'regression':
    #     prediction_test = tester.test_regressor(dataset_test)

    # result = '\t'.join(map(str, [prediction_test]))
    # print(result)
    # print('forward_regressor:', model.forward_regressor(list(zip(*dataset_test)), train = False))

    # vec = pp.preprocess_single_mol(task, ['C=CC(=O)Nc1ccc2ncnc(Nc3ccc(F)c(Br)c3)c2c1'], torch.Tensor([0.31109404580972566]), radius, device)
    # vec = list(zip(*vec))
    # print('vec:', model.forward_regressor(vec, train = False))
