# ChemTS - Lite

This is a on-going renewal project of the original ChemTS code. We aim to:
  1. Make the code easier to understand and easier to use
  2. Add python3 and TensorFlow 2.0 support
  3. Make the evaluation function easy to modify

Molecule Design using Monte Carlo Tree Search with Neural Rollout. ChemTS can design novel molecules with desired properties(such as, HOMO-LUMO gap, energy, logp..). Combining with rDock, ChemTS can design molecules active to target proteins. The ChemTS paper is available at https://arxiv.org/abs/1710.00616 .　Also, we introduced the distributed parallel ChemTS that can accerlate molecular discovery. And the distributed parallel ChemTS is available at https://github.com/tsudalab/DP-ChemTS.

#  Requirements 
1. [Python](https://www.anaconda.com/download/)>=2.7 Python 3 supported
2. [Keras](https://github.com/fchollet/keras) 
3. [rdkit](https://anaconda.org/rdkit/rdkit)

#  How to use ChemTS? 
##  Train a RNN model for molecule generation
1. cd train_RNN
2. Modify the main function according to your needs
>* Input file name as a training set.(Input file format is SMILES)
>* Save model name.
>* Set epochs based on the training set size. (Early stopping by default)
3. Run python train_RNN.py to train the RNN model. GPU is highly recommended for reducing the training time.

## Predict molecule by RNN model
1. cd ChemTS
2. Modify the main function according to your needs
>* Input the trained RNN model file. 
>* Modify evaluation function here. (Logp evaluation by default)
>```Python
>valid_compound = UCTchemical(evaluate_node_logp)
>```
3. Run python mcts_logp.py


# License
This package is distributed under the MIT License.
