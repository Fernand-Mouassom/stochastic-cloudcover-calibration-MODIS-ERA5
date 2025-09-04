import numpy as np
import pandas as pd
from scipy.stats import entropy
import xarray as xr
from scipy.interpolate import interp1d
from scipy.ndimage import generic_filter
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import griddata
from sklearn.base import BaseEstimator
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform
from scipy.stats import skew
from sklearn.metrics import make_scorer
from utils import fogModel

data_file = "./grid_search_data.csv" # Change with the correct path
data = pd.read_csv(data_file)

nxf = 14 # change according to the size of your data
nyf = 25 # change according to the size of your data
xf=np.linspace(0, 14, nxf)
yf=np.linspace(0, 25, nyf)
Xf, Yf=np.meshgrid(xf, yf)

def sigma_initial1(nyf, nxf):
    target_percent_ones= np.random.choice([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],p = [0.14,0.08,0.08,0.08,0.08,0.08,0.08,0.08,0.08,0.08,0.14])
    percent_ones = np.random.uniform(max(0, target_percent_ones), min(1, target_percent_ones))
    total_elements = nyf * nxf
    num_ones = int(round(percent_ones * total_elements))
    num_zeros = total_elements - num_ones

    matrix = np.array([1] * num_ones + [0] * num_zeros)
    np.random.shuffle(matrix)
    l = matrix.reshape(nyf, nxf)
    return l

dt = 0.5
taus0=2*3600
taut0=10*3600
tau=2*60
qs0=0.007
tauf=qs0/(2*3600)
sigma0=0.5
D0=10
b=D0*np.ones((nyf, nxf))
r=1.3 # change if needed
h0=30 # change if needed

def FogModel_new(N, mean_q, qs, tau, nxf, nyf, dt, h0, r, J0):
    model_dist = []
    for ii in range(N):
        sigma_model = sigma_initial1(nyf, nxf)
        sigmaM = []
        for _ in range(2000):
            sigma_model = fogModel(sigma_model, mean_q, qs, tau, nxf, nyf, dt, h0, r, J0)
            sigmaM.append(np.mean(sigma_model))
        model_CAF = np.mean(sigmaM[-500:])
        model_dist.append(model_CAF)
    return model_dist

class FogModelEstimator(BaseEstimator):
    def __init__(self, J0=1):
        self.J0 = J0
    def fit(self, X, y):
        return self
    def predict(self, X):
        predicted_CAF = []
        for q_i, qs_i in X:
            sigma_final = FogModel_new(N, q_i, qs_i, tau, nxf, nyf, dt, h0, r, self.J0)
            caf_pred = np.mean(sigma_final)
            predicted_CAF.append(caf_pred)
        return np.array(predicted_CAF)


def composite_score(y_true, y_pred, a1=0.5, a2=0.25, a3=0.25):
    mse = np.abs(np.mean(y_true) - np.mean(y_pred))/np.mean(y_true)
    std = np.abs(np.std(y_true) - np.std(y_pred))/np.std(y_true)
    skw = np.abs(skew(y_true) - skew(y_pred))/np.abs(skew(y_true))
    score = a1*mse + a2*std + a3*skw
    return score

def composite_ms(y_true, y_pred, a1=0.5, a2=0.5):
    mse = np.abs(np.mean(y_true) - np.mean(y_pred))/np.mean(y_true)
    std = np.abs(np.std(y_true) - np.std(y_pred))/np.std(y_true)
    mse_std= a1*mse + a2*std
    return mse_std

def mse_only(y_true, y_pred):
    return np.abs(np.mean(y_true) - np.mean(y_pred))/np.mean(y_true)

def std_only(y_true, y_pred):
    return  np.abs(np.std(y_true) - np.std(y_pred))/np.std(y_true)

def skew_only(y_true, y_pred):
    return np.abs(skew(y_true) - skew(y_pred))/np.abs(skew(y_true))

scoring = {
    'score': make_scorer(composite_score, greater_is_better=False),
    'mse_std': make_scorer(composite_ms, greater_is_better=False),
    'mse': make_scorer(mse_only, greater_is_better=False),
    'std': make_scorer(std_only, greater_is_better=False),
    'skew': make_scorer(skew_only, greater_is_better=False),
}

num_bins_q = 1 # change if needed
num_bins_T = [288.35856501, 290.17734426, 290.78360401, 291.38986376, 291.79403692, 292.19821009, 292.60238325, 292.90551312, 293.208643, 293.81490275, 294.2695975625 , 295.02742225, 295.633682, 297.45246124] # change if needed

data['q_bin'] = pd.cut(data['q'], bins=num_bins_q)
data['T_bin'] = pd.cut(data['T'], bins=num_bins_T)

bin_stats = data.groupby(['q_bin', 'T_bin']).agg(
    mean_CAF=('CAF', 'mean'),
    std_CAF=('CAF', 'std'),
    count=('CAF', 'count')
).reset_index()


locs=[2.74, 3.22, 3.49, 3.51, 3.85, 3.96, 4.00, 4.02, 4.16, 4.22, 4.43, 4.66, 4.76] # change if needed
scales=[0.8 , 0.13, 0.4 , 0.39, 0.23, 0.14, 0.2 , 0.28, 0.18, 0.25, 0.14, 0.18, 0.16] # change if needed

all_results = []

for i in range(len(bin_stats['T_bin'].unique())):
    selected_T_bin = bin_stats['T_bin'].unique()[i]
    subset = data[data['T_bin'] == selected_T_bin]
    N=200 # change if needed
    qs_bin = np.array(subset['qs'])
    CAF_bin = np.array(subset['CAF'])

    block_size = 70 # change if needed
    if i==0:
        block_size = 20

    if i in [8, 9, 10]:
        block_size = 180

    n_blocks = len(qs_bin) // block_size

    qs_flat = qs_bin[:n_blocks * block_size]
    qs_blocks = qs_flat.reshape(-1, block_size)

    CAF_flat = CAF_bin[:n_blocks * block_size]
    CAF_blocks = CAF_flat.reshape(-1, block_size)

    qs_means = qs_blocks.mean(axis=1)
    q_means = np.array(data['q'].mean())*np.ones(len(qs_means))
    CAF_means = CAF_blocks.mean(axis=1)

    X = np.vstack([q_means, qs_means]).T
    y = CAF_means

    param_dist = {
        'J0': uniform(loc=locs[i], scale=scales[i])
    }

    fog_model = FogModelEstimator()

    search = RandomizedSearchCV(
        fog_model,
        param_distributions=param_dist,
        n_iter=70, # change if needed
        scoring=scoring,
        refit=False,
        cv=3,
        verbose=1,
        n_jobs=-1
    )
    search.fit(X, y)


    results = search.cv_results_
    results_df = pd.DataFrame({
        'T_bin': str(subset['T_bin'].values[0]),
        'J0': [p['J0'] for p in results['params']],
        'mean_test_mse': -results['mean_test_mse'],
        'mean_test_std': -results['mean_test_std'],
        'mean_test_skew': -results['mean_test_skew'],
        'mean_test_score': -results['mean_test_score'],
        'mean_test_mse_std': -results['mean_test_mse_std'] 
    })


    all_results.append(results_df)


final_df = pd.concat(all_results, ignore_index=True)
output_path = "h30n_gridsearch70_results.csv" # Change with the correct path
final_df.to_csv(output_path, index=False)
