import pandas as pd
import numpy as np
from scipy import stats as scistats
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from scipy.sparse import csr_matrix
from sklearn.cross_validation import train_test_split
from CTM import CTM
from CTMParallel import CTMParallel
import csv
import sys
import pickle

if len(sys.argv) != 2:
	print "Usage: RunCTM_ratorgans.py numtopics"
	sys.exit()
data = pd.read_csv('/n/regal/scrb152/Data/Yu_et_al/full_counts_matrix.csv').set_index('GeneID')
genes_wanted = (data > 1).sum(axis=1) > 5
genes_wanted = data.var(axis=1).sort(inplace=False, ascending=False)[:1000].index
counts_newsetup = data.ix[genes_wanted, :].T
classes = np.array(map(lambda x: x.split("_")[0], counts_newsetup.index))
vocab = counts_newsetup.columns
counts = counts_newsetup.values.astype(float)

numTopicsWanted = int(sys.argv[1])
numProcesses = 64
print "Pickling data structures"
pickle.dump(vocab, open("/n/home09/ankitgupta/281FinalProj/pickled_objects/rat/vocab_rat_" + sys.argv[1] + "topics.p", "wb"))
pickle.dump(classes, open("/n/home09/ankitgupta/281FinalProj/pickled_objects/rat/classes_rat_" + sys.argv[1] + "topics.p", "wb"))
pickle.dump(counts, open("/n/home09/ankitgupta/281FinalProj/pickled_objects/rat/counts_rat_" + sys.argv[1] + "topics.p", "wb"))
print "Running CTM..."
ctm = CTMParallel(counts.shape[0], numTopicsWanted, counts.shape[1], numProcesses,counts, 10, .001)
ctm.EM()
pickle.dump(ctm, open("/n/home09/ankitgupta/281FinalProj/pickled_objects/rat/model_rat_" + sys.argv[1] + "topics.p", "wb"))
