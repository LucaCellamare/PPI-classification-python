

import pandas as pd 
import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
from sklearn.metrics import accuracy_score, f1_score



graph_name = "ppi"


def build_dataframe(input_data: pd.DataFrame, col_name: str, preserve_int_col_name=False) -> pd.DataFrame:
    """
    Given an input DataFrame and a column name, return a new DataFrame in which the column has been cleaned.
    Used to transform features and labels columns from "0;1;1;0" to [0, 1, 1, 0]
    """
   
    vertices_dict = []
    for i, row_i in input_data.iterrows():
        features = [int(float(x)) for x in row_i[f"{col_name}s"].split(";")]
        
        new_v = {"id": i}
        for j, f in enumerate(features):
            new_v[j if preserve_int_col_name else f"{col_name}_{j}"] = f
        vertices_dict += [new_v]
    res_df = pd.DataFrame(vertices_dict)
    return res_df.set_index("id")


def build_vertices():
    # Read vertex features and classes in the training set;
    vertices_path = f"./data/{graph_name}_train.csv"
    vertices_train = pd.read_csv(vertices_path, sep=",", index_col="id")
    pd.DataFrame(vertices_train).to_csv("vertices_train.csv")
    vertices_train["dataset"] = "train"

    # Read vertex features in the test/validation set;
    vertices_path = f"./data/{graph_name}_test.csv"
    vertices_test = pd.read_csv(vertices_path, sep=",", index_col="id")
    vertices_test["dataset"] = "test"

    return vertices_train,vertices_test


def build_graph():
    edges_path = f"./data/{graph_name}_e.csv"
    Data = open(edges_path, "r")
    next(Data, None)  # skip the first line in the input file
    Graphtype = nx.Graph()
    G = nx.parse_edgelist(Data, delimiter=',', create_using=Graphtype,
                      nodetype=int)
    G.remove_edges_from(G.selfloop_edges())  # removing self-loops
    return G


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """
    Convert a scipy sparse matrix to a torch sparse tensor.
    """
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def adjacency_matrix_GCN(G, theta=1):
    #  BUILDING ADJ MATRIX FOR GCN
    A = nx.to_scipy_sparse_matrix(G)
    A = A+A.T
    A += theta*sp.eye(A.shape[0])  #  Â = A + I
    A = sp.coo_matrix(A)  # sparse matrix in coordinate format
    rowsum = np.array(A.sum(1))  # D
    D = np.power(rowsum, -0.5).flatten()  # D = D^(-1/2)
    D[np.isinf(D)] = 0.
    D = sp.diags(D)
    A.dot(D).transpose().dot(D).tocoo()
    return sparse_mx_to_torch_sparse_tensor(A)


def edge_list_SAGE():
    #  CREATING EDGES LIST FOR SAGE
    A = pd.read_csv('./data/ppi_e.csv')
    return A.values.transpose()


def hamming_accuracy(prediction, true_values):
    """
    Metric used in multi-label classification,
    for each example measures the % of correctly predicted labels.
    
    Equivalent to traditional accuracy in a single-output scenario;
    """
    return np.mean(np.sum(np.equal(prediction, true_values)) / float(true_values.size))


def get_score(prediction, true_values):
    print("\tHamming accuracy: {:.3f}".format(hamming_accuracy(prediction, true_values)))
    print("\tAccuracy, exact matches: {:.3f}".format(accuracy_score(prediction, true_values)))
    print("\tMacro F1 Score: {:.3f}".format(f1_score(y_true=true_values, y_pred=prediction, average="macro")))
    print("\tMicro F1 Score: {:.3f}".format(f1_score(y_true=true_values, y_pred=prediction, average="micro")))

    
    
def bool_to_int(labels: list) -> list:
    """
    Turn a list of 0s and 1s into a list whose values are the indices of 1s.
    Used to create a valid Kaggle submission.
    E.g. [1, 0, 0, 1, 1] -> [0, 3, 4]
    """
    return [i for i, x in enumerate(labels) if x == 1]
    

def get_results(filename,prediction,X_test_df):
    y_pred = [" ".join([str(y) for y in bool_to_int(x)]) for x in prediction]
    y_pred_df = pd.DataFrame(y_pred, columns=["labels"], index=X_test_df.index)
    y_pred_df.to_csv(filename)




def a_third_law(labels,p):
    counter = np.zeros(labels.shape[0])
    for i in range(labels.shape[0]):
        for j in range(labels[i].shape[0]):
            if labels[i][j] == 1:
                counter[j] += 1
    for i in range(counter.shape[0]):
        if counter[i] > labels.shape[0]/3 :
            counter[i] = 1
        else :
            counter[i] = 0
    
    for col in range(counter.shape[0]):
        for row in range(p.shape[0]):
            if (counter[col] == 1 and p[row][col] != 1) :
                p[row][col] = 1
    return p
        

def get_lmean(labels):
    lmean = np.zeros((labels.shape[1],2))
    for i in range(labels.shape[1]):
        for j in range(labels.shape[0]):
                lmean[i][0] += labels[j][i]
        lmean[i][0] = lmean[i][0]
        lmean[i][1] = i
    return lmean


def sort_lmean(lmean):
    lmean = lmean[lmean[:,0].argsort()]  
    lmean_f = np.zeros(lmean.shape)
    for i in range(lmean.shape[0]):
        for j in range(lmean.shape[1]):
            lmean_f[i][j] = lmean[121-i][j]
    return lmean_f