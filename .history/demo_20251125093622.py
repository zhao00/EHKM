import numpy as np
import time
from utils import loadmat, evaluate_performance
from EHKM import EHKM
# from pynndescent import NNDescent     ## https://github.com/lmcinnes/pynndescent

data_path  = './dataset/'
dataset_list = ['Letter','USPS']

for index,data_name in enumerate(dataset_list):
    
    print("We are now clustering the "+data_name+" dataset.")

    X, y_true, N, dim, k = loadmat("{}{}.mat".format(data_path,data_name))
    X = X.astype(np.float64)

    start_time = time.time()
    c, num_clusters, req_c, time_list = EHKM(X, req_clust=k, verbose=False, return_knn_time= True)
    # c, num_clusters, req_c = EHKM(X, req_clust=k, verbose=False, return_knn_time= False)
    end_time = time.time()

    if req_c.ndim == 1:
        y_pred = req_c.reshape(1,-1)[0]
    else:
        y_pred = req_c[:,-1]

    acc, nmi, purity = evaluate_performance(y_true, y_pred)
    print(f'{data_name:<10}:\t{acc:.3f}\t{nmi:.3f}\t{purity:.3f}\t{end_time-start_time:.2f}')