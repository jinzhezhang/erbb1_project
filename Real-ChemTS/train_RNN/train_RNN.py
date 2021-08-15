from keras.models import Sequential
from keras.layers import Dense, TimeDistributed
from keras.layers import GRU
from keras.layers.embeddings import Embedding
from keras.optimizers import Adam
from keras.layers import Dropout
import numpy as np
from keras.utils.np_utils import to_categorical
from keras.preprocessing import sequence
from keras import callbacks
import os
import csv
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def prepare_data(smiles, all_smile):
    all_smile_index = []
    for i in range(len(all_smile)):
        smile_index = []
        for j in range(len(all_smile[i])):
            smile_index.append(smiles.index(all_smile[i][j]))
        all_smile_index.append(smile_index)
    X_train = all_smile_index
    Y_train = []
    for i in range(len(X_train)):
        x1 = X_train[i]
        x2 = x1[1:len(x1)]
        x2.append(0)
        Y_train.append(x2)

    return X_train, Y_train


def save_model(modelFile, model):
    # serialize model to JSON
    model_json = model.to_json()
    with open(modelFile+".json", "w") as f:
        f.write(model_json)
    # serialize weights to HDF5
    model.save_weights(modelFile+".h5")
    print("Saved model to disk")


def save_val(val):
    fp = open("../data/train_val.txt", "w")
    fp.write(str(val))
    fp.close()


def read_smiles(trainingFile):
    smiles_list = []
    f = open(trainingFile, 'r')
    reader = csv.reader(f)
    for row in reader:
        smiles_list.append(row)
    f.close()
    smiles_list_processed = []
    for i in range(len(smiles_list)):
        word1 = smiles_list[i]
        smiles_list_processed.append(word1[0])
    return smiles_list_processed


def smiles_processed(ori_smiles_list):
    all_smile = []
    end = "$"
    val = ["$"]
    element_table = ["C", "N", "B", "O", "P", "S", "F", "Cl", "Br", "I", "(", ")", "=", "#"]

    for i in range(len(ori_smiles_list)):
        smiles = ori_smiles_list[i]
        smiles_list_processed = []
        j = 0
        while j < len(smiles):
            smiles_space = []
            if smiles[j] == "[":
                smiles_space.append(smiles[j])
                j = j + 1
                while smiles[j] != "]":
                    smiles_space.append(smiles[j])
                    j = j + 1
                smiles_space.append(smiles[j])
                smiles_space_list = ''.join(smiles_space)
                smiles_list_processed.append(smiles_space_list)
                j = j + 1
            else:
                smiles_space.append(smiles[j])

                if j + 1 < len(smiles):
                    smiles_space.append(smiles[j + 1])
                    smiles_space_list = ''.join(smiles_space)
                else:
                    smiles_space.insert(0, smiles[j - 1])
                    smiles_space_list = ''.join(smiles_space)

                if smiles_space_list not in element_table:
                    smiles_list_processed.append(smiles[j])
                    j = j + 1
                else:
                    smiles_list_processed.append(smiles_space_list)
                    j = j + 2

        smiles_list_processed.append(end)
        smiles_list_processed.insert(0, "&")
        all_smile.append(list(smiles_list_processed))
    for i in range(len(all_smile)):
        for j in range(len(all_smile[i])):
            if all_smile[i][j] not in val:
                val.append(all_smile[i][j])

    return val, all_smile


if __name__ == "__main__":
    trainingFile = "../data/erbb1_ligand.csv"
    modelFile="../train_RNN/erbb1"
    smiles = read_smiles(trainingFile)
    val, all_smile = smiles_processed(smiles)
    save_val(val)
    print(val)
    val_len = len(val)
    X_train, Y_train = prepare_data(val, all_smile)

    maxlen = 81
    X = sequence.pad_sequences(X_train, maxlen=maxlen, dtype='int32',
                               padding='post', truncating='pre', value=0.)
    Y = sequence.pad_sequences(Y_train, maxlen=maxlen, dtype='int32',
                               padding='post', truncating='pre', value=0.)

    Y_train_one_hot = np.array([to_categorical(sent_label, num_classes=len(val)) for sent_label in Y])

    model = Sequential()

    model.add(Embedding(input_dim=val_len, output_dim=val_len, input_length=X.shape[1], mask_zero=False))
    model.add(GRU(units=256, return_sequences=True, activation="tanh", input_shape=(maxlen, 64)))
    model.add(Dropout(0.2))
    model.add(GRU(256, activation='tanh', return_sequences=True))
    model.add(Dropout(0.2))
    model.add(TimeDistributed(Dense(val_len, activation='softmax')))
    optimizer = Adam(lr=0.01)
    print(model.summary())
    early_stop = callbacks.EarlyStopping(monitor='val_acc', min_delta=0.002, patience=2, mode='max')
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    model.fit(X, Y_train_one_hot, batch_size=512, epochs=150, validation_split=0.1, callbacks=[early_stop])

    save_model(modelFile, model)
