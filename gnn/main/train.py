import sys
import timeit
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle as pkl
import preprocess as pp
import config
from sklearn.metrics import roc_auc_score
#from models import MolecularGraphNeuralNetwork, Trainer, Tester
import models

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

    print('Preprocessing the', config.dataset, 'dataset.')
    print('Just a moment......')
    (dataset_train, dataset_dev, dataset_test,
     N_fingerprints) = pp.create_datasets(config.task, config.dataset,
     config.radius, config.device)
    print('-'*100)

    print('The preprocess has finished!')
    print('# of training data samples:', len(dataset_train))
    print('# of development data samples:', len(dataset_dev))
    print('# of test data samples:', len(dataset_test))
    print('-'*100)

    print('Creating a model.')
    torch.manual_seed(1234)
    model = models.MolecularGraphNeuralNetwork(
            N_fingerprints, config.dim, config.layer_hidden,
            config.layer_output).to(config.device)
    trainer = models.Trainer(model)
    tester = models.Tester(model)
    print('# of model parameters:',
          sum([np.prod(p.size()) for p in model.parameters()]))
    print('-'*100)

    file_result = '../output/result--' + "test" + '.txt'
    if config.task == 'classification':
        result = 'Epoch\tTime(sec)\tLoss_train\tAUC_train\tAUC_test'
    if config.task == 'regression':
        result = 'Epoch\tTime(sec)\tLoss_train\tMAE_train\tMAE_test'

    with open(file_result, 'w') as f:
        f.write(result + '\n')

    print('Start training.')
    print('The result is saved in the output directory every epoch!')

    np.random.seed(1234)

    start = timeit.default_timer()

    for epoch in range(config.iteration):

        epoch += 1
        if epoch % config.decay_interval == 0:
            trainer.optimizer.param_groups[0]['lr'] *= config.lr_decay

        loss_train = trainer.train(dataset_train)

        if config.task == 'classification':
            prediction_dev = tester.test_classifier(dataset_dev)
            prediction_test = tester.test_classifier(dataset_test)
        if config.task == 'regression':
            prediction_dev = tester.test_regressor(dataset_dev)
            prediction_test = tester.test_regressor(dataset_test)

        time = timeit.default_timer() - start

        if epoch == 1:
            minutes = time * config.iteration / 60
            hours = int(minutes / 60)
            minutes = int(minutes - 60 * hours)
            print('The training will finish in about',
                  hours, 'hours', minutes, 'minutes.')
            print('-'*100)
            print(result)

        result = '\t'.join(map(str, [epoch, time, loss_train,
                                     prediction_dev, prediction_test]))
        tester.save_result(result, file_result)

        print(result)

    torch.save(model.state_dict(), '../model/'+str(config.iteration)+'_gnn.pth')
