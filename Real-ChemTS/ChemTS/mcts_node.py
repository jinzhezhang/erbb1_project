import numpy as np
from keras.preprocessing import sequence
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import MolFromSmiles
import sascorer
import networkx as nx
from rdkit.Chem import rdmolops
import sys
sys.path.append('../..')
from gnn.main.predict import predict_vec
import gnn.main.config as config
import gnn.main.preprocess as pp


def expand_node(model, state, val):
    all_nodes = []
    position = []
    position.extend(state)
    get_index = []
    for j in range(len(position)):
        #print ("val.index(position[j])1:", val.index(position[j]))
        get_index.append(val.index(position[j]))

    get_index = get_index

    x = np.reshape(get_index, (1, len(get_index)))
    x_pad = sequence.pad_sequences(x, maxlen=81, dtype='int32',
                                   padding='post', truncating='pre', value=0.)

    for i in range(30):
        predictions = model.predict(x_pad)
        preds = np.asarray(predictions[0][len(get_index) - 1]).astype('float64')
        preds = np.log(preds) / 1.0
        preds = np.exp(preds) / np.sum(np.exp(preds))
        next_probas = np.random.multinomial(1, preds, 1)
        next_int = np.argmax(next_probas)
        all_nodes.append(next_int)

    all_nodes = list(set(all_nodes))
    return all_nodes


def add_node(all_nodes, val):
    added_nodes = []
    for i in range(len(all_nodes)):
        added_nodes.append(val[all_nodes[i]])
    return added_nodes


def simulate_node(model, state, val, added_nodes):
    all_posible = []
    maxlen = 81
    end = "$"
    for i in range(len(added_nodes)):
        position = []
        position.extend(state)
        position.append(added_nodes[i])
        total_generated = []
        get_index = []
        for j in range(len(position)):
            get_index.append(val.index(position[j]))

        x = np.reshape(get_index, (1, len(get_index)))
        x_pad = sequence.pad_sequences(x, maxlen=maxlen, dtype='int32',
                                       padding='post', truncating='pre', value=0.)
        while not get_index[-1] == val.index(end):
            predictions = model.predict(x_pad)
            preds = np.asarray(predictions[0][len(get_index) - 1]).astype('float64')
            preds = np.log(preds) / 1.0
            preds = np.exp(preds) / np.sum(np.exp(preds))
            next_probas = np.random.multinomial(1, preds, 1)
            next_int = np.argmax(next_probas)
            get_index.append(next_int)
            x = np.reshape(get_index, (1, len(get_index)))
            x_pad = sequence.pad_sequences(x, maxlen=maxlen, dtype='int32',
                                           padding='post', truncating='pre', value=0.)
            if len(get_index) > maxlen:
                break
        total_generated.append(get_index)
        all_posible.extend(total_generated)
    return all_posible


def predict_smile(all_posible, val):
    new_compound = []
    for i in range(len(all_posible)):
        total_generated = all_posible[i]
        generate_smile = []
        for j in range(len(total_generated) - 1):
            generate_smile.append(val[total_generated[j]])
        generate_smile.remove("&")
        new_compound.append(generate_smile)
    return new_compound


def make_smile(generate_smile):
    new_compound = []
    for i in range(len(generate_smile)):
        middle = []
        for j in range(len(generate_smile[i])):
            middle.append(generate_smile[i][j])
        com = ''.join(middle)
        new_compound.append(com)
    return new_compound


def evaluate_erbb1_log50(gnn_model, dicts, new_compound):
    node_index = []
    valid_compound = []
    scores = []
    for i in range(len(new_compound)):
        try:
            m = Chem.AddHs(Chem.MolFromSmiles(str(new_compound[i])))
        except:
            print('None')
            m = None
            with open('ChemTS_erbb1_2000_optimized_score.txt','a') as f:
                f.write(str(new_compound[i]) + '| None'  + '\n')
        if m and len(new_compound[i]) <= 81:
            node_index.append(i)
            valid_compound.append(new_compound[i])
            try:
                res = predict_vec(gnn_model,'regression',
                '../../gnn/dataset/regression/erbb1_clean_log_ic50/',
                 1, 'erbb1_clean_log_ic50', config.device, [new_compound[i]],
                 dicts)
            except:
                res = [99]
            with open('ChemTS_erbb1_2000_optimized_score.txt','a') as f:
                f.write(str(new_compound[i]) + '|' + str(-res[0]) + '\n')
            scores.append(max(5-res[0], 0))
    return node_index, scores, valid_compound

def evaluate_node_logp(new_compound):
    node_index = []
    valid_compound = []
    score = []
    logP_values = np.loadtxt('logP_values.txt')
    logP_mean = np.mean(logP_values)
    logP_std = np.std(logP_values)
    cycle_scores = np.loadtxt('cycle_scores.txt')
    cycle_mean = np.mean(cycle_scores)
    cycle_std = np.std(cycle_scores)
    SA_scores = np.loadtxt('SA_scores.txt')
    SA_mean = np.mean(SA_scores)
    SA_std = np.std(SA_scores)

    for i in range(len(new_compound)):
        try:
            m = Chem.AddHs(Chem.MolFromSmiles(str(new_compound[i])))
        except:
            print ('None')
        if m != None and len(new_compound[i]) <= 81:
            try:
                logp = Descriptors.MolLogP(m)
            except:
                logp = -1
            node_index.append(i)
            valid_compound.append(new_compound[i])
            SA_score = -sascorer.calculateScore(MolFromSmiles(new_compound[i]))
            cycle_list = nx.cycle_basis(nx.Graph(rdmolops.GetAdjacencyMatrix(MolFromSmiles(new_compound[i]))))
            if len(cycle_list) == 0:
                cycle_length = 0
            else:
                cycle_length = max([len(j) for j in cycle_list])
            if cycle_length <= 6:
                cycle_length = 0
            else:
                cycle_length = cycle_length - 6
            cycle_score = -cycle_length
            SA_score_norm = (SA_score - SA_mean) / SA_std
            logp_norm = (logp - logP_mean) / logP_std
            cycle_score_norm = (cycle_score - cycle_mean) / cycle_std
            score_one = SA_score_norm + logp_norm + cycle_score_norm
            score.append(score_one)
    return node_index, score, valid_compound
