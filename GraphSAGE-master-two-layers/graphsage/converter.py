import pandas as pd 
import numpy as np
data = np.load('val.npy')
pd.DataFrame(data).to_csv("path/to/val1.csv")