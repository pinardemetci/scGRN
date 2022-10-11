import glob
import os
import pickle

import numpy as np
import pandas as pd
from arboreto.algo import grnboost2
from arboreto.utils import load_tf_names
from ctxcore.rnkdb import FeatherRankingDatabase as RankingDatabase
from dask.diagnostics import ProgressBar
from pyscenic.aucell import aucell
from pyscenic.prune import df2regulons, prune2df
from pyscenic.utils import load_motifs, modules_from_adjacencies


def save_obj(obj, name):
    with open('data/obj/' + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open('data/obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

if __name__ ==  '__main__':
    DATA_FOLDER="/gpfs/data/rsingh47/hzaki1/atacseqdataP0/resources"
    RESOURCES_FOLDER="/gpfs/data/rsingh47/hzaki1/atacseqdataP0/resources"
    MM_TFS_FNAME = os.path.join(RESOURCES_FOLDER, 'mm_mgi_tfs.txt')
    SC_EXP_FNAME = os.path.join(DATA_FOLDER, "P0_exp_matrix.tsv")

    ex_matrix = pd.read_csv(SC_EXP_FNAME, sep='\t', header=0, index_col=0)
    

    tf_names = load_tf_names(MM_TFS_FNAME)

    adjacencies = grnboost2(ex_matrix, gene_names=ex_matrix.columns,tf_names=tf_names, verbose=True)

    adjacencies.to_csv('/gpfs/data/rsingh47/hzaki1/atacseqdataP0/resources/adjacencies.tsv', sep='\t', header=True, index=False)

