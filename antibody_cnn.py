import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Reshape
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier


def classification(classifier_type, antibody, another_antibody, encode_method='blosum'):
    all = pd.read_csv('./samples/all_ab_pre_post.txt', delimiter='\t')

    #shuffle the samples

    all = all.sample(frac=1).reset_index(drop=True)

    max_k = all.CDR3K.str.len().max()
    max_h = all.CDR3H.str.len().max()

    all['paddedk'] = all["CDR3K"].str.pad(width=max_k, fillchar="-", side="right")
    all['paddedh'] = all["CDR3H"].str.pad(width=max_h, fillchar="-", side="right")
    all['padded'] = all['paddedk'] + all['paddedh']

    #filter the duplicate sequences,
    all2 = all.sort_values("post", ascending=False)
    all2 = all2.drop_duplicates(subset=["padded"], keep="first")

    all2 = all2.sample(frac=1).reset_index(drop=True)

    #not filter the duplicate sequences
    all2 = all

    #define the binder/non biner
    #The results of this step is different from the original code for four rows. That is because of some elements with similar sequences has similar post but different pre.
    conditions = [
        (all2['post'] >= 0.01) & (all2['fc'] >= 1.8),
        (all2['pre'] >= 0.01) & (all2['fc'] < 1),
        ~(((all2['post'] >= 0.01) & (all2['fc'] >= 1.8)) | ((all2['pre'] >= 0.01) & (all2['fc'] < 1)))
        ]

    values = ['1', '0', 'ambi']
    all2['enriched'] = np.select(conditions, values)
    all2 = all2[all2['enriched'] != 'ambi']

    def get_encoded(seq, encode_mat):
        if len(seq) == 0:
            raise ValueError('Sequence has zero characters')
        res = np.zeros((len(seq), len(encode_mat)))
        for i, c in enumerate(seq):
            res[i] = encode_mat[c]
        return res

    def convert_seqs_to_mat(seqs, encode_mat):
        if len(seqs) == 0:
            raise ValueError('No sequences')
        seq_size = len(seqs[0])
        for seq in seqs:
            if len(seq) != seq_size:
                raise ValueError('Sequences of Different Size')
        res = np.zeros((len(seqs), seq_size, len(encode_mat)))
        for i, seq in enumerate(seqs):
            res[i] = get_encoded(seq, encode_mat)
        return res

    def make_balanced_df(antibody_df):
        df_p = antibody_df[antibody_df['enriched'] == '1']
        df_n = antibody_df[antibody_df['enriched'] == '0']
        ratio = float(len(df_n))/len(df_p)
        if ratio < 1:
            raise ValueError('dataset has more positive cases than negative!')
        dups = []
        for i in range(int(ratio)):
            dups.append(df_p.copy())
        df_p = pd.concat(dups)
        diff = len(df_n) - len(df_p)
        df_p = pd.concat([df_p, df_p.sample(n=diff)])
        return pd.concat([df_p, df_n]).sample(frac=1)



    if encode_method == 'blosum':
        encode_mat = pd.read_csv('./samples/blosum.csv', index_col=0)
    elif encode_method == 'onehot':
        encode_mat = pd.read_csv('./samples/onehot.txt', index_col=0, sep='\t')

    antibody_df = all2[all2['antigen'] == antibody]
    antibody_df = make_balanced_df(antibody_df)
    X = convert_seqs_to_mat(list(antibody_df['padded']), encode_mat)
    X = np.expand_dims(X, axis=3)
    Y = pd.to_numeric(antibody_df['enriched'], downcast='integer')


    def convert_binary_to_onehot(Y):
        res = np.zeros((Y.size, 2))
        res[np.arange(Y.size), Y] = 1
        return res


    Y = convert_binary_to_onehot(Y)

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=32)

    if classifier_type == 'cnn':
        model = Sequential()
        model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=(36, 22, 1), padding="same"))
        model.add(Dropout(0.5))
        model.add(MaxPooling2D(pool_size=(2, 2), padding="same"))
        model.add(Conv2D(32, kernel_size=(3, 4), activation='relu', padding="same"))
        model.add(Dropout(0.4))
        model.add(MaxPooling2D(pool_size=(2, 2), padding="same"))
        model.add(Conv2D(32, kernel_size=(4, 4), activation='relu', padding="same"))
        model.add(Dropout(0.3))
        model.add(Flatten())
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.4))
        model.add(Dense(2, activation='softmax'))
        model.compile(loss=keras.losses.categorical_crossentropy, optimizer=Adam(), metrics=['accuracy'])
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=32)
        model.fit(x_train, y_train, batch_size=50, epochs=32, verbose=0, validation_data=(x_val, y_val))
    elif classifier_type == 'rf':
        model = RandomForestClassifier(random_state=0, n_estimators=50, warm_start=True, n_jobs=-1)
        nsamples, nx, ny, nz = x_train.shape
        xx = x_train.reshape((nsamples, nx*ny))
        model.fit(xx, y_train)
        antibody_df = all2[all2['antigen'] == another_antibody]
        X = convert_seqs_to_mat(list(antibody_df['padded']), encode_mat)
        X = np.expand_dims(X, axis=3)
        Y = pd.to_numeric(antibody_df['enriched'], downcast='integer')
        Y = convert_binary_to_onehot(Y)
        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=32)
        nsamples, nx, ny, nz = x_test.shape
        x_test = x_test.reshape((nsamples, nx*ny))

    y_pred = model.predict(x_test)

    np.argmax(y_pred, axis=1)
    acc = accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))
    f1 = f1_score(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))
    precision = precision_score(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))
    return acc, f1, precision, encode_method
