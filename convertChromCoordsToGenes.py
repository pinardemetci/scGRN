import os
import pickle
import re

import pandas as pd
from numpy.lib.npyio import save
from pyensembl import EnsemblRelease, genome
from pyensembl.species import human
from tqdm import tqdm

DATA_FOLDER="/gpfs/data/rsingh47/hzaki1/atacseqdataChromatin/"
RESOURCES_FOLDER="/gpfs/data/rsingh47/hzaki1/atacseqdataChromatin/resources"
SC_EXP_FNAME = os.path.join(RESOURCES_FOLDER, "GSE126074_CellLineMixture_SNAREseq_cDNA_counts.tsv")
COUNTS_FNAME = os.path.join(RESOURCES_FOLDER, "GSE126074_CellLineMixture_SNAREseq_chromatin_counts.tsv")

reads = pd.read_csv(COUNTS_FNAME, sep='\t')
expression = pd.read_csv(SC_EXP_FNAME, sep='\t', header=0, index_col=0).T

data = EnsemblRelease(species=human)

data.download()
data.index()

geneList = expression.columns.values.tolist()


def save_obj(obj, name):
    with open('/gpfs/data/rsingh47/hzaki1/atacseqdataChromatin/obj/' + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open('/gpfs/data/rsingh47/hzaki1/atacseqdataChromatin/obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

def range_overlapping(x, y):
    if x.start == x.stop or y.start == y.stop:
        return False
    return x.start <= y.stop and y.start <= x.stop

geneCoordDict = {}
counter = 0
for gene in geneList:
    # geneCap = gene.capitalize()
    try:
        lookups = data.genes_by_name(gene)
    except:
        counter+=1
        continue
    for geneLookup in lookups:
        chromosome = geneLookup.contig
        start = geneLookup.start
        end = geneLookup.end
        if chromosome not in geneCoordDict:
            geneCoordDict[chromosome] = {}
        geneCoordDict[chromosome][range(start, end)] = gene

# counter = 0
# geneSet = set()
# rangeSet = set()
# for chromosome in geneCoordDict:
#     for generange in geneCoordDict[chromosome]:
#         rangeSet.add(generange)
#         geneSet.add(geneCoordDict[chromosome][generange])

# print(len(geneSet))
# print(len(rangeSet))
# print(counter)

counter =0
regions = list(reads.index)
resultsDict = {}
for region in tqdm(regions, total=len(regions)):
    matches = re.match("chr(.*):(.*)-(.*)", region)
    chromosome, start, end = matches[1], int(matches[2]), int(matches[3])
    if chromosome not in geneCoordDict:
        counter +=1
        continue
    for geneRegion in geneCoordDict[chromosome]:
        if range_overlapping(geneRegion, range(start+10000,end+10000)):
            if geneCoordDict[chromosome][geneRegion] not in resultsDict:
                resultsDict[geneCoordDict[chromosome][geneRegion]] = [region]
            else:
                resultsDict[geneCoordDict[chromosome][geneRegion]].append(region)

save_obj(resultsDict, "regionToGene_human10")

