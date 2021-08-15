from math import *
import numpy as np
import random as pr
from keras.models import model_from_json
from ast import literal_eval
from mcts_node import simulate_node, make_smile, predict_smile, add_node, expand_node, evaluate_node_logp, evaluate_erbb1_log50
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
sys.path.append('/home/zhang/elix')
import gnn.main.preprocess as pp
import gnn.main.config as config
import gnn.main.models as models
import torch
import pickle


class Chemical:
    def __init__(self):
        self.position = ['&']

    def Clone(self):
        st = Chemical()
        st.position = self.position[:]
        return st

    def SelectPosition(self, m):
        self.position.append(m)


class Node:
    def __init__(self, position=None, parent=None, state=None):
        self.position = position
        self.parentNode = parent
        self.childNodes = []
        self.child = None
        self.wins = 0
        self.visits = 0
        self.depth = 0

    def Selectnode(self):
        ucb = []
        for i in range(len(self.childNodes)):
            score = self.childNodes[i].wins / self.childNodes[i].visits + 1.0 * sqrt(
                2 * log(self.visits) / self.childNodes[i].visits)
            ucb.append(float('%.2f' % score))
        m = np.amax(ucb)
        indices = np.nonzero(ucb == m)[0]
        ind = pr.choice(indices)
        s = self.childNodes[ind]
        return s

    def Addnode(self, m, s):
        n = Node(position=m, parent=self, state=s)
        self.childNodes.append(n)

    def Update(self, result):
        self.visits += 1
        self.wins += result



def MCTS(root,evaluation_function, gnn_model, dicts):
    """initialization of the chemical trees and grammar trees"""
    rootnode = Node(state=root)
    maxnum = 0

    """global variables used for save valid compounds and simulated compounds"""
    valid_compound = []
    seen_compound = []
    max_score = float('-inf')
    current_score = []
    all_score = []
    depth = []
    best_score = float('-inf')
    resultsFile = "../data/prediction_record.csv"

    while maxnum < 2000:
        print("iteration:",maxnum)
        node = rootnode
        state = root.Clone()
        """selection step"""
        node_pool = []
        while node.childNodes != []:
            node= node.Selectnode()
            state.SelectPosition(node.position)

        depth.append(len(state.position))
        if len(state.position) >= 81:
            re = -1.0
            while node != None:
                node.Update(re)
                node = node.parentNode
        else:
            """expansion step"""
            """calculate how many nodes will be added under current leaf"""
            expanded_node = expand_node(model, state.position, val)
            added_node = add_node(expanded_node, val)

            all_posible = simulate_node(model, state.position, val, added_node)
            generate_smile = predict_smile(all_posible, val)
            new_compounds = make_smile(generate_smile)

            if new_compounds not in seen_compound:
                node_index, score, smiles = evaluation_function(gnn_model, dicts,
                new_compounds)
                seen_compound.extend(new_compounds)
                print ("smiles:", smiles,"score:", score, 'best_so_far:', best_score)
                save_record_to_file(resultsFile, state.position, smiles, score)
                if score and max(score) > best_score:
                    best_score = max(score)
                if len(node_index) == 0:
                    re = -1.0
                    while node != None:
                        node.Update(re)
                        node = node.parentNode
                else:
                    re = []
                    for i in range(len(node_index)):
                        m = node_index[i]
                        maxnum = maxnum + 1
                        node.Addnode(added_node[m], state)
                        node_pool.append(node.childNodes[i])
                        if score[i] >= max_score:
                            max_score = score[i]
                            current_score.append(max_score)
                        else:
                            current_score.append(max_score)
                        depth.append(len(state.position))
                        re.append((0.8 * score[i]) / (1.0 + abs(0.8 * score[i])))
                    for i in range(len(node_pool)):
                        node = node_pool[i]
                        while node != None:
                            node.Update(re[i])
                            node = node.parentNode
            else:
                re = -1.0
                while node != None:
                    node.Update(re)
                    node = node.parentNode


    # print ("max score found:", current_score)
    # print ("num_valid:", len(valid_compound))
    # print ("valid_compound:", valid_compound)
    return valid_compound


def UCTchemical(evaluation_function, gnn_model, dicts):
    state = Chemical()
    best = MCTS(root=state,evaluation_function=evaluation_function,
                gnn_model = gnn_model, dicts = dicts)
    return best


def loaded_model(modelFile):
    json_file = open(modelFile+'.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(modelFile+'.h5')
    print("Loaded model from disk")
    return loaded_model


def save_record_to_file(resultsFile, prefix, smiles, logp):
    of = open(resultsFile, 'a')
    prefix.remove('&')
    print ("prefix:" + "".join(prefix))
    print ("smiles:" + ",".join(smiles))
    print ("score:" + ",".join('%s' % id for id in logp) + "\n")
    of.write("prefix:" + "".join(prefix) + "\n")
    of.write("smiles:" + ",".join(smiles) + "\n")
    of.write ("score:" + ",".join('%s' % id for id in logp) + "\n" + "\n")
    of.close()

def save_predictions_to_file(outFile, valid_compound):
    fp = open(outFile, "w")
    fp.write('\n'.join(valid_compound))
    fp.close()


def read_val():
    fp = open("../data/train_val.txt", "r")
    val_str = fp.read()
    fp.close()
    return literal_eval(val_str)

def load_gnn_model(path, device):
    model = torch.load(path)
    model.to(device)
    return model

if __name__ == "__main__":
    val = ['$', '&', 'Br', '.', 'C', '(', 'N', 'c', '1', 'n', '2', '[nH]', '-', '3', 'O', ')', 'F', '[C@@H]', '4', 's', '=', 'Cl', 'o', '5', 'S', '/', '#', '[C@H]', '[N+]', '[O-]', '[S+]', '\\', '6', 'P', '[C@]', 'I', '[n+]', '[C@@]', '[O]', '7', 'B', '[Br-]', '[Se]', '[S@+]', '[Cl-]', '[Si]', '[N-]', '[Na+]', '[o+]', '[Zn+2]']
    #val = ['$', '&', 'O', 'C', 'c', '1', '(', 'N', ')', '=', '#', '[C@H]', 's', '[C@@H]', 'S', 'n', '/', '\\', 'o', 'P', '2', '[C]', '[N]', 'Cl', '[nH]', '[n]', 'p', '[Si]', '[C@@]', '[P@@]', '[S]', '[NH]', 'F', '[P@]', 'Br', '[SiH2]', '[O-]', '[C@]', '[O]', '[NH2]', '-', '[CH]', '[n+]', '[S@]', 'B', '3', '[P]', '[Se]', 'I', '[SeH]', '[nH+]', '[N+]', '[SiH3]']
    #val = []
    #val = read_val()
    modelFile = "../train_RNN/erbb1"
    predictionFile = "erbb1_predictions_2000.txt"
    atom_file = 'atom_dict.pkl'
    bond_file = 'bond_dict.pkl'
    fp_file = 'fp_dict.pkl'
    edge_file = 'edge_dict.pkl'
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
    gnn_model = load_gnn_model('../../gnn/model/200_gnn.pth', config.device)
    model = loaded_model(modelFile)
    valid_compound = UCTchemical(evaluate_erbb1_log50, gnn_model, dicts)

    save_predictions_to_file(predictionFile, valid_compound)
