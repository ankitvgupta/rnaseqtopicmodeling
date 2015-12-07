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

totrain = "all"
if len(sys.argv) != 2:
	print "Usage: RunCTM.py numTopics"
	sys.exit()

print totrain
features =  pd.read_table("/n/home09/ankitgupta/281FinalProj/GSE60361_C1-3005-Expression.txt", sep='\t', usecols=range(3006)).set_index('cell_id').T
print features.shape
lines = []
with open("/n/home09/ankitgupta/281FinalProj/expression_mRNA_17-Aug-2014.txt", 'r') as f:
    reader = csv.reader(f, delimiter='\t')
    for line in reader:
        lines.append(line)
metadata = pd.DataFrame(lines)
metadata = metadata.set_index(0)
metadata.columns = metadata.ix['cell_id', :]
#metadata.columns = metadata.ix[7, :]
#metadata.rename(columns={'cell_id':'info'}).set_index('info')
classification = metadata.ix['group #', :]

# This contains the classification for each gene into 9 major classes of genes.

genes_of_interest = (-features.sum()).sort(inplace=False)[20:].index
remove_low_expression_genes = features.sum() > 100
genes_wanted = remove_low_expression_genes[remove_low_expression_genes].index.intersection(genes_of_interest)
genes_wanted = features.T.var(axis=1).sort(inplace=False, ascending=False)[:1000].index.intersection(genes_wanted)
#genes_of_interest = genes_of_interest & remove_low_expression_genes
#print features.index
#print classification[classification == totrain].index
print "Number of genes being investigated", len(genes_wanted)
features = features.ix[:, genes_wanted]
classification = classification[features.index]
X_train, X_test, y_train, y_test, samples_train, samples_test = train_test_split(features, classification, features.index, train_size=500, test_size=100, random_state=42)
# We can compare the classification problem using standard SVM to that with LDA, and later with Pachinko Allocation.
vocab = features.columns
#print compressed_features.shape

numTopicsWanted = int(sys.argv[1])
numProcesses = 64
#counts = np.array(compressed_features.todense())
#print tmp.shape
#print tmp
print "Pickling data structures"
pickle.dump(vocab, open("/n/home09/ankitgupta/281FinalProj/pickled_objects/vocab_neuron_" + sys.argv[1] + "topics.p", "wb"))
pickle.dump(X_train, open("/n/home09/ankitgupta/281FinalProj/pickled_objects/counts_neuron_" + sys.argv[1] + "topics.p", "wb"))
pickle.dump(y_train, open("/n/home09/ankitgupta/281FinalProj/pickled_objects/classes_neuron_" + sys.argv[1] + "topics.p", "wb"))
print "Running CTM..."
ctm = CTMParallel(X_train.shape[0], numTopicsWanted, X_train.shape[1], numProcesses, X_train, 12, .001)
ctm.EM()
pickle.dump(ctm, open("/n/home09/ankitgupta/281FinalProj/pickled_objects/model_neuron_" + sys.argv[1] + "topics.p", "wb"))
