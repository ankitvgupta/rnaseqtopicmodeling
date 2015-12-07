import pandas as pd
import numpy as np
from scipy import stats as scistats
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from scipy.sparse import csr_matrix
from sklearn.cross_validation import train_test_split
from CTM import CTM
import csv
import sys
import pickle

data = pd.read_csv('/n/regal/scrb152/Data/Yu_et_al/full_counts_matrix.csv').set_index('GeneID')
genes_wanted = (data > 1).sum(axis=1) > 5
genes_wanted = data.var(axis=1).sort(inplace=False, ascending=False)[:1000].index
counts_newsetup = data.ix[genes_wanted, :].T
classes = np.array(map(lambda x: x.split("_")[0], counts_newsetup.index))
vocab = counts_newsetup.columns
counts = counts_newsetup.values.astype(float)

numTopicsWanted = 10
numProcesses = 64
print "Pickling data structures"
pickle.dump(vocab, open("/n/home09/ankitgupta/281FinalProj/pickled_objects/vocabratserial.p", "wb"))
pickle.dump(classes, open("/n/home09/ankitgupta/281FinalProj/pickled_objects/classesratserial.p", "wb"))
pickle.dump(counts, open("/n/home09/ankitgupta/281FinalProj/pickled_objects/countsratserial.p", "wb"))
print "Running CTM..."
ctm = CTM(counts.shape[0], numTopicsWanted, counts.shape[1], counts, 5, .001)
ctm.EM()
pickle.dump(ctm, open("/n/home09/ankitgupta/281FinalProj/pickled_objects/modelratserial.p", "wb"))
