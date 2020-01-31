import argparse
import csv
import pandas as pd

from tqdm import tqdm
from utilities import *

# Get Tau_h and Tau_l from input
parser = argparse.ArgumentParser(description = "Script used to calibrate NAUTICA parameters.")
parser.add_argument('tau_h',type=float,
    help='Threshold used in the non-interaction, non-biogrid case')
parser.add_argument('tau_l',type=float,
    help='Threshold used in the interaction, non-biogrid case')

def nautica_calibration(tau_h, tau_l):
    # paths and variables
    TR = 'data/validation/tr.csv'
    TS = 'data/validation/ts.csv'
    BIOGRID_FILE = 'data/BIOGRID-MV-Physical-3.4.162.tab2.txt'
    PRED_CSV_FILE = 'data/tica_03/TICA_predictions.csv'
    TF_FILE = 'data/human_tfs.tsv'

    MIN_DEGREE = 3  # If less than this degree, a node is discarded from prediction
    TAXON_NMBR = 9606  # Used to filter non-human data

    # Syntactic sugar to make network access clearer
    # Build BIOGRID and TICA networks
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

    # Here, "algo" represents either TICA or 
    # any other interaction prediction algorithm

    tf_algo_predictions = dict()
    with open(PRED_CSV_FILE, 'r') as pred_file:
        pred_f_reader = csv.reader(pred_file)
        for row in pred_f_reader:
            i = (row[0], row[1]) if row[0] <= row[1] else (row[1],row[0])
            if i not in tf_algo_predictions.keys():
                tf_algo_predictions[i] = 1 if row[-1] == 'True' else 0
            else:
                tf_algo_predictions[i] = 1 if row[-1] == 'True' or tf_algo_predictions[i] == 1 else 0

    tf_algo_prots = sorted(list(set([i[0] for i in tf_algo_predictions]).union(set([i[1] for i in tf_algo_predictions]))))

    algo = AlgoNetwork("TICA_30",tf_algo_prots,tf_algo_predictions)  # change accordingly
    algo.get_tfs(tfs)
    algo.build_graph()

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
            # s_prot_net = biogrid.graph.subgraph(nx.common_neighbors(biogrid.graph,p1,p2))
            # density = 2*len(biogrid.graph.subgraph(p1p2_neighbors).edges) / (n_p1p2 * (n_p1p2 -1)) if n_p1p2 >1 else -1 # Density of a network of  n nodes is defined by # of observed edges divided (n *(n-1) / 2).
            #print(density)
            is_in_biogrid = 1 if (p1,p2) in biogrid.interactions else 0
            tr_dataset.loc[index,'Is_Biogrid'] = is_in_biogrid
            is_predicted_by_algo = -1 if (p1,p2) not in algo.interactions else algo.interactions[(p1,p2)]
            tr_dataset.loc[index,'Int_pred'] = is_predicted_by_algo
            nautica_result = nautica_predictor(n_p1p2,is_in_biogrid,is_predicted_by_algo,tau_h,tau_l)
            tr_dataset.loc[index,'nautica_prediction'] = nautica_result
    #print(f"Amount of predictions found: {len(tr_dataset[~tr_dataset['nautica_prediction'].isna()])}")

    # Computing quality measures
    training_measures = ""
    for l in ['COOP','COMP','NINT']:
        true_pos = len(tr_dataset[(tr_dataset['LABEL'] == l) & (tr_dataset['nautica_prediction'] == l)])
        false_pos = len(tr_dataset[(tr_dataset['LABEL'] != l) & (tr_dataset['nautica_prediction'] == l)])
        true_neg = len(tr_dataset[(tr_dataset['LABEL'] != l) & (tr_dataset['nautica_prediction'] != l)])
        false_neg = len(tr_dataset[(tr_dataset['LABEL'] == l) & (tr_dataset['nautica_prediction'] != l)])
        # print(f"For class {l}, here is the confusion matrix:")
        # print("CM\tPN\tPP")
        # print(f"AN\t{true_neg}\t{false_pos}")
        # print(f"AP\t{false_neg}\t{true_pos}")
        # print()
        training_measures += f"{tau_h},{tau_l},{l},{true_pos/(true_pos+false_neg)},{true_pos/(true_pos+false_pos)},{true_neg/(true_neg+false_pos)}"
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
            # s_prot_net = biogrid.graph.subgraph(nx.common_neighbors(biogrid.graph,p1,p2))
            # density = 2*len(biogrid.graph.subgraph(p1p2_neighbors).edges) / (n_p1p2 * (n_p1p2 -1)) if n_p1p2 >1 else -1 # Density of a network of  n nodes is defined by # of observed edges divided (n *(n-1) / 2).
            #print(density)
            is_in_biogrid = 1 if (p1,p2) in biogrid.interactions else 0
            ts_dataset.loc[index,'Is_Biogrid'] = is_in_biogrid
            is_predicted_by_algo = -1 if (p1,p2) not in algo.interactions else algo.interactions[(p1,p2)]
            ts_dataset.loc[index,'Int_pred'] = is_predicted_by_algo
            nautica_result = nautica_predictor(n_p1p2,is_in_biogrid,is_predicted_by_algo,tau_h,tau_l)
            ts_dataset.loc[index,'nautica_prediction'] = nautica_result
    #print(f"Amount of predictions tested: {len(ts_dataset[~ts_dataset['nautica_prediction'].isna()])}")

    # Computing quality measures
    testing_measures = ""
    for l in ['COOP','COMP','NINT']:
        true_pos = len(ts_dataset[(ts_dataset['LABEL'] == l) & (ts_dataset['nautica_prediction'] == l)])
        false_pos = len(ts_dataset[(ts_dataset['LABEL'] != l) & (ts_dataset['nautica_prediction'] == l)])
        true_neg = len(ts_dataset[(ts_dataset['LABEL'] != l) & (ts_dataset['nautica_prediction'] != l)])
        false_neg = len(ts_dataset[(ts_dataset['LABEL'] == l) & (ts_dataset['nautica_prediction'] != l)])
        # print(f"For class {l}, here is the confusion matrix:")
        # print("CM\tPN\tPP")
        # print(f"AN\t{true_neg}\t{false_pos}")
        # print(f"AP\t{false_neg}\t{true_pos}")
        # print()
        testing_measures += f"{tau_h},{tau_l},{l},{true_pos/(true_pos+false_neg)},{true_pos/(true_pos+false_pos)},{true_neg/(true_neg+false_pos)}"
        testing_measures += "\n" if l != 'NINT' else ""
    return (training_measures,testing_measures)



if __name__ == '__main__':
    args = parser.parse_args()
    (training_measures,testing_measures) = nautica_calibration(args.tau_h,args.tau_l)
    # output is inteded to be written to a csv
    # header: TAU_h, TAU_l, LABEL, recall, precision, specificity
    print(f"train: {training_measures}")
    print(f"test: {testing_measures}")
