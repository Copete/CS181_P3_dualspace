## P3: Classifying Sounds
## Antonio Copete

import numpy as np
import librosa
import time
from sklearn.preprocessing import scale

SAMPLE_RATE = 22050

def get_features(X, sample_rate=SAMPLE_RATE):
    '''
    Functions to extract features from array of amplitudes
    '''
    stft = np.abs(librosa.stft(X)) # Short-time Fourier Transform
    mfcc = librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40)
    mfcc_delta  = librosa.feature.delta(mfcc, width=9, order=1)
    mfcc_delta2 = librosa.feature.delta(mfcc, width=9, order=2)
    chroma = librosa.feature.chroma_stft(S=stft, sr=sample_rate)
    mel = librosa.feature.melspectrogram(X, sr=sample_rate)
    contrast = librosa.feature.spectral_contrast(S=stft, sr=sample_rate)
    tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate)
    return mfcc, mfcc_delta, mfcc_delta2, chroma, mel, contrast, tonnetz

def get_feature_set(file, sample_rate=SAMPLE_RATE, training_set=True):
    '''
    Get full feature set from file
    '''
    print('File: ', file)
    chunksize = 100
    n_chunks = 0
    time0 = time.clock()
    ids, labels, mfcc, mfcc_delta, mfcc_delta2 = None, None, None, None, None
    chroma, mel, contrast, tonnetz = None, None, None, None
    for df in pd.read_csv(file, header=None, index_col=(None if training_set else 0), chunksize=chunksize):
        n_chunks += 1
        print('Chunk #',n_chunks,':')
        if training_set:
            X = df.values[:,:-1]
            labels0 = df.values[:,-1].reshape(-1,1).astype(int)
            labels = labels0 if n_chunks==1 else np.vstack((labels, labels0))
        else:
            X = df.values
            ids0 = df.index.values.reshape(-1,1).astype(int)
            ids = ids0 if n_chunks==1 else np.vstack((ids, ids0))
        for X0 in X:
            mfcc0, mfcc_delta0, mfcc_delta20, chroma0, mel0, contrast0, tonnetz0 = get_features(X0, sample_rate=SAMPLE_RATE)
            mfcc = [mfcc0] if mfcc is None else np.vstack((mfcc, [mfcc0]))
            mfcc_delta  = [mfcc_delta0]  if mfcc_delta  is None else np.vstack((mfcc_delta,  [mfcc_delta0]))
            mfcc_delta2 = [mfcc_delta20] if mfcc_delta2 is None else np.vstack((mfcc_delta2, [mfcc_delta20]))
            chroma = [chroma0] if chroma is None else np.vstack((chroma, [chroma0]))
            mel = [mel0] if mel is None else np.vstack((mel, [mel0]))
            contrast = [contrast0] if contrast is None else np.vstack((contrast, [contrast0]))
            tonnetz = [tonnetz0] if tonnetz is None else np.vstack((tonnetz, [tonnetz0]))
        print('   So far: {} samples, {:.2f} min'.format(len(mfcc), (time.clock()-time0)/60))
    print('DONE with full feature set for file', file)
    return ids, labels, mfcc, mfcc_delta, mfcc_delta2, chroma, mel, contrast, tonnetz

# Training set
_, y_train, X_train_mfcc, X_train_mfcc_delta, X_train_mfcc_delta2, X_train_chroma, X_train_mel, X_train_contrast, X_train_tonnetz = get_feature_set('../train.csv.gz', sample_rate=SAMPLE_RATE, training_set=True)

# Test set
X_test_ids, _, X_test_mfcc, X_test_mfcc_delta, X_test_mfcc_delta2, X_test_chroma, X_test_mel, X_test_contrast, X_test_tonnetz = get_feature_set('../test.csv.gz', sample_rate=SAMPLE_RATE, training_set=False)

# Concatenate and compress them into time-averaged features
get_t_avg = lambda X: np.array([np.mean(X0,axis=1) for X0 in X])
get_t_std = lambda X: np.array([np.std(X0,axis=1)  for X0 in X])
X_train = np.hstack((get_t_avg(X_train_mfcc), get_t_avg(X_train_mfcc_delta),
                     get_t_avg(X_train_mfcc_delta2),
                     get_t_avg(X_train_chroma), get_t_avg(X_train_mel),
                     get_t_avg(X_train_contrast), get_t_avg(X_train_tonnetz)))
X_test = np.hstack((get_t_avg(X_test_mfcc), get_t_avg(X_test_mfcc_delta),
                    get_t_avg(X_test_mfcc_delta2),
                    get_t_avg(X_test_chroma), get_t_avg(X_test_mel),
                    get_t_avg(X_test_contrast), get_t_avg(X_test_tonnetz)))
print(X_train.shape, X_test.shape)
N_train = X_train.shape[0]
# Scale full feature set (skip next 3 lines for no scaling)
X_scl = scale(np.vstack((X_train,X_test)))
X_train, X_test = X_scl[:N_train], X_scl[N_train:]
del X_scl

# Transform y_train into 1-hot array
Y_train = np.zeros((N_train,10),dtype=int)
Y_train[np.arange(N_train),y_train[:,0]] = 1
