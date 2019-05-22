import numpy as np
import utils  
#from node2vec import Node2Vec  # NODE2VEC EMBEDDINGS 
import classifier as clf  # VERTEX CLASSIFIER
import matplotlib.pyplot as plt  # PLOTTING
import torch
from gcn.model import GCN 
from train  import train_model 


# LOADING THE DATA
vertices_train, vertices_test = utils.build_vertices()

X_train_df = utils.build_dataframe(vertices_train, "feature")
#X_train_df = X_train_df.drop(['feature_10'], axis=1)  # DROP USELESS FEATURE

X_test_df = utils.build_dataframe(vertices_test, "feature")
#X_test_df = X_test_df.drop(['feature_10'], axis=1)  # DROP USELESS FEATURE

labels_df = utils.build_dataframe(vertices_train, "label", preserve_int_col_name=True)


# BUILDING NUMPY MATRICES 
X_train = X_train_df.values
X_test = X_test_df.values
X = np.concatenate((X_train, X_test), axis=0)
labels = labels_df.values

# MASK FOR TRAIN/TEST
train_idx = range(X_train.shape[0])
test_idx = range(X_train.shape[0],X.shape[0])

# BUILDING THE GRAPH
G = utils.build_graph() # ALSO REMOVES SELF-LOOPS
G = G.to_directed() 

# BUILDING ADJ MATRIX FOR GCN
A = utils.adjacency_matrix_GCN(G, theta=1)  


# DEFINING OUR PARAMETERS
n_features = X.shape[1]
n_classes = labels.shape[1]
n_hidden = 32  #  NUMBER OF HIDDEN PARAMETERS IN OUR NET
# CREATING AND TRAINING OUR MODEL

gcn_model = GCN(n_features, n_hidden, n_classes, dropout=0.5)

embedding_gcn = train_model(gcn_model, X, A, labels, train_idx, epochs=150, lr=0.005, wd=5e-3)

#  EXTRACTING EMBEDDINGS 
train_embedding_gcn = embedding_gcn[train_idx]
test_embedding_gcn = embedding_gcn[test_idx]


#  TESTING TRAIN ACCURACY (SIGMOID TRESHOLD SET AT 0.4 )
gcn_train_pred = torch.sigmoid(train_embedding_gcn).detach().numpy() > 0.4 
utils.get_score(gcn_train_pred, labels)  # HAMMING ACCURACY, F1-MICRO, F1-MACRO

#  SAVING THE EMBEDDING
np.save('./embedding/embedding_gcn.npy', embedding_gcn.detach().numpy())  # .NPY FILE

torch.save(model.state_dict(),'/home/users/michele.bertoldi/test_gcn/gcn_tizi_train.pt')
#  MAKING OUR PREDICTION
gcn_test_pred = torch.sigmoid(test_embedding_gcn).detach().numpy() > 0.5  # PREDICTIONS ON TEST SET
gcn_test_pred = utils.a_third_law(labels,gcn_test_pred) # 0.475 accuracy
utils.get_results('./results/gcn_pred.csv',gcn_test_pred, X_test_df)  # SAVING RESULTS IN A .CSV 


