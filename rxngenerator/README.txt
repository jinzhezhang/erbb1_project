0) Requirements
- Linux or MacOS (We run our experiments on Linux server and MacOs)
- RDKit (version='2020.09.5')
- Python (version=3.9.4)
- Pytorch (version=1.8.0)


1) Training
To train the model, type the following command:
python trainvae.py -w 200 -l 50 -d 2 -v "data.txt_fragmentvocab.txt" -t "data.txt"

The weights of the trained model will be saved in the folder "weights", which will be loaded to run sampling and Bayesian Optimization.

2) To sample new molecules with traiend model, simpy run:
python sample.py -w 200 -l 50 -d 2 -v "weights/data.txt_fragmentvocab.txt" -t "data.txt" -s "weights/uspto_vae_iter-100.npy" -o “Results/generated_rxns.txt"

The generated moleucles and associated reaction trees will be saved in file: "Results/generated_rxns.txt"

3) To run Bayesian optimization, simpy run:
python run_bo.py -w 200 -l 50 -d 2 -r 1 -v “weights/data.txt_fragmentvocab.txt" -t "data.txt" -s "weights/uspto_vae_iter-100.npy" -m “qed”

Please change the parameter -r with different radom seed numbers. We performed 10 times of running BO, which results in 10 files of valid reaction trees saved in the folder Results. To see reaction trees with top optimized QED scores, simply type:

python print_results.py 


