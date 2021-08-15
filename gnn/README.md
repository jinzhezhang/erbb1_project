## GNN based ligand acitivity preidiction

This work is forked from: https://github.com/masashitsubaki/molecularGNN_smiles

In the current version, the following changes has been made:

- Created functions to save, load and predict with loaded model. 
- Corrected a dictionary leak bug which will allow training process to “see” testing data. Without fixing this bug, the model can only predict accurately if the testing set remains the same order than they are during training. 
- Put all global configs into config.py so that they are easy to mangage and declare only once per execution. 
- Made the package importable.
- Corrected a bug where original code omit first line of dataset
- Re-structured the code into 4 files： train, predict, model, preprocess



## Cite the original work

```
@article{tsubaki2018compound,
  title={Compound-protein Interaction Prediction with End-to-end Learning of Neural Networks for Graphs and Sequences},
  author={Tsubaki, Masashi and Tomii, Kentaro and Sese, Jun},
  journal={Bioinformatics},
  year={2018}
}
```
