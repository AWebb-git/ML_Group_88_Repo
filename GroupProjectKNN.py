import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sklearn 
from sklearn.metrics import mean_squared_error,r2_score,f1_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import model_selection
from sklearn.model_selection import KFold
import matplotlib.patches as mpatches
from sklearn.metrics import roc_curve
from sklearn.neighbors import KNeighborsClassifier