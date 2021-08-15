# erbb1_project

This project uses 3 generative models: RXN Generator, JT-VAE and ChemTS, and 1 predictive model: GNN, to generate candidates with good affinity against erbb1 receptor.

# Requirements

# Usage

For usage of each model, please refer to the corresponding ipynb file.

# Credits:

RXN Generator code is inspired by the orginal work of Dai Hai Nguyen:

https://github.com/haidnguyen0909/rxngenerator
https://arxiv.org/pdf/2106.03394

JT-VAE is created by Wengong Jin:
https://arxiv.org/pdf/1802.04364.pdf

The implementation of JT-VAE is inspired by:
https://github.com/Bibyutatsu/FastJTNNpy3


ChemTS is a molecule generator created by Yang et al. in 2017:
https://arxiv.org/abs/1710.00616


GNN predictive model is inspired by Tsubaki et al.'s work:

https://academic.oup.com/bioinformatics/article/35/2/309/5050020?login=true
https://github.com/masashitsubaki/molecularGNN_smiles

All modifications and other code are created by: Jinzhe Zhang
