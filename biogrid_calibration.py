import argparse
import csv
import pandas as pd

from tqdm import tqdm
from utilities import *

# Get Tau_h and Tau_l from input
parser = argparse.ArgumentParser(description = "Script used to calibrate parameters of a biogrid-based decision tree.")
parser.add_argument('h',type=float,
    help='Threshold used in the non-interaction, non-biogrid case')
parser.add_argument('l',type=float,
    help='Threshold used in the interaction, non-biogrid case')

def bg_dt_calibration(H, L):
    # paths and variables
    TR = 'data/validation/tr.csv'
    TS = 'data/validation/ts.csv'
    BIOGRID_FILE = 'data/BIOGRID-MV-Physical-3.4.162.tab2.txt'
    TF_FILE = 'data/human_tfs.tsv'

    MIN_DEGREE = 3  # If less than this degree, a node is discarded from prediction
    TAXON_NMBR = 9606  # Used to filter non-human data

    # Syntactic sugar to make network access clearer
    # Build BIOGRID networks
    biogrid_interactions = []
    with open(BIOGRID_FILE, 'r') as biogrid_infile:
        bg_reader = csv.reader(biogrid_infile, delimiter='\t')
        for row in bg_reader:
            if row[15] == str(TAXON_NMBR) and row[16] == str(TAXON_NMBR):
                biogrid_interactions.append((row[7], row[8]) if row[7] <= row[8] else (row[8],row[7]))  # Int1, Int2, PID

    bg_prots = sorted(list(set([i[0] for i in biogrid_interactions]).union(set([i[1] for i in biogrid_interactions]))))

    biogrid = Network("BioGRID",bg_prots,biogrid_interactions)

    tfs = set()
    with open(TF_FILE, 'r') as tf_infile:
        tf_reader = csv.reader(tf_infile, delimiter='\t')
        for row in tf_reader:
            tfs.add(row[0])

    biogrid.get_tfs(tfs)
    biogrid.build_graph()

    # Get training and testing datasets
    tr_dataset = pd.read_csv(TR, sep=',',index_col=False,header=0)
    print(f"Size of training dataset: {len(tr_dataset)}")
    ts_dataset = pd.read_csv(TS, sep=',',index_col=False,header=0)

    # run nautica using input threshold on TR
    for index,row in tqdm(tr_dataset.iterrows()):
        p1 = row[0]
        p2 = row[1]
        p1_neighbors = set(biogrid.graph[p1]) if p1 in biogrid.graph else []
        p2_neighbors = set(biogrid.graph[p2]) if p2 in biogrid.graph else []
        if len(p1_neighbors) >= MIN_DEGREE and len(p2_neighbors) >= MIN_DEGREE:
            n_p1p2 = len(p1_neighbors.intersection(p2_neighbors))
            tr_dataset.loc[index,'N_12'] = n_p1p2
            # for reference
            is_in_biogrid = 1 if (p1,p2) in biogrid.interactions else 0
            tr_dataset.loc[index,'Is_Biogrid'] = is_in_biogrid

            result = 'COOP' if n_p1p2 >= H else 'COMP' if n_p1p2 >= L else 'NINT'
            tr_dataset.loc[index,'bg_dt_prediction'] = result
    #print(f"Amount of predictions found: {len(tr_dataset[~tr_dataset['nautica_prediction'].isna()])}")

    # Computing quality measures
    training_measures = ""
    for l in ['COOP','COMP','NINT']:
        true_pos = len(tr_dataset[(tr_dataset['LABEL'] == l) & (tr_dataset['bg_dt_prediction'] == l)])
        false_pos = len(tr_dataset[(tr_dataset['LABEL'] != l) & (tr_dataset['bg_dt_prediction'] == l)])
        true_neg = len(tr_dataset[(tr_dataset['LABEL'] != l) & (tr_dataset['bg_dt_prediction'] != l)])
        false_neg = len(tr_dataset[(tr_dataset['LABEL'] == l) & (tr_dataset['bg_dt_prediction'] != l)])
        # print(f"For class {l}, here is the confusion matrix:")
        # print("CM\tPN\tPP")
        # print(f"AN\t{true_neg}\t{false_pos}")
        # print(f"AP\t{false_neg}\t{true_pos}")
        # print()
        training_measures += f"{H},{L},{l},{true_pos/(true_pos+false_neg)},{true_pos/(true_pos+false_pos)},{true_neg/(true_neg+false_pos)}"
        training_measures += "\n" if l != 'NINT' else ""

    #print(f"Size of testing dataset: {len(tr_dataset)}")
    
    # run nautica using input threshold on TS
    for index,row in tqdm(ts_dataset.iterrows()):
        p1 = row[0]
        p2 = row[1]
        p1_neighbors = set(biogrid.graph[p1]) if p1 in biogrid.graph else []
        p2_neighbors = set(biogrid.graph[p2]) if p2 in biogrid.graph else []
        if len(p1_neighbors) >= MIN_DEGREE and len(p2_neighbors) >= MIN_DEGREE:
            n_p1p2 = len(p1_neighbors.intersection(p2_neighbors))
            ts_dataset.loc[index,'N_12'] = n_p1p2
            # for reference
            is_in_biogrid = 1 if (p1,p2) in biogrid.interactions else 0
            ts_dataset.loc[index,'Is_Biogrid'] = is_in_biogrid
            
            result = 'COOP' if n_p1p2 >= H else 'COMP' if n_p1p2 >= L else 'NINT'
            ts_dataset.loc[index,'bg_dt_prediction'] = result
    #print(f"Amount of predictions tested: {len(ts_dataset[~ts_dataset['nautica_prediction'].isna()])}")

    # Computing quality measures
    testing_measures = ""
    for l in ['COOP','COMP','NINT']:
        true_pos = len(ts_dataset[(ts_dataset['LABEL'] == l) & (ts_dataset['bg_dt_prediction'] == l)])
        false_pos = len(ts_dataset[(ts_dataset['LABEL'] != l) & (ts_dataset['bg_dt_prediction'] == l)])
        true_neg = len(ts_dataset[(ts_dataset['LABEL'] != l) & (ts_dataset['bg_dt_prediction'] != l)])
        false_neg = len(ts_dataset[(ts_dataset['LABEL'] == l) & (ts_dataset['bg_dt_prediction'] != l)])
        # print(f"For class {l}, here is the confusion matrix:")
        # print("CM\tPN\tPP")
        # print(f"AN\t{true_neg}\t{false_pos}")
        # print(f"AP\t{false_neg}\t{true_pos}")
        # print()
        testing_measures += f"{H},{L},{l},{true_pos/(true_pos+false_neg)},{true_pos/(true_pos+false_pos)},{true_neg/(true_neg+false_pos)}"
        testing_measures += "\n" if l != 'NINT' else ""
    return (training_measures,testing_measures)



if __name__ == '__main__':
    args = parser.parse_args()
    (training_measures,testing_measures) = bg_dt_calibration(args.h,args.l)
    # output is inteded to be written to a csv
    # header: TAU_h, TAU_l, LABEL, recall, precision, specificity
    print(f"train: {training_measures}")
    print(f"test: {testing_measures}")