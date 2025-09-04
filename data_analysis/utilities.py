import numpy as np
import pandas as pd
from scipy.stats import entropy
import xarray as xr
from scipy.interpolate import interp1d
from scipy.ndimage import generic_filter
import matplotlib.pyplot as plt
import xarray as xr
import seaborn as sns
from scipy.interpolate import griddata
from sklearn.base import BaseEstimator
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform
from scipy.stats import skew
from sklearn.metrics import make_scorer
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from scipy.optimize import curve_fit

def linear_fit(x, y):
    x = np.array(x)
    y = np.array(y)
    
    # Reshape for sklearn
    X_features = x.reshape(-1, 1)
    y_reshaped = y.reshape(-1, 1)
    
    # Build and train model
    model = LinearRegression()
    model.fit(X_features, y_reshaped)
    
    # Get the coefficients and intercept
    coefficients = model.coef_[0]  
    intercept = model.intercept_[0]  # Get the intercept

    # Generate smooth x values for plotting the fitted line
    x_plot = np.linspace(x.min(), x.max(), 100).reshape(-1, 1)
    
    # Predict y values using the linear model
    y_pred = model.predict(x_plot)

    print(f'Power Law parameters: \n K : {10**intercept} \n alpha : {coefficients}')
    
    return y_pred, x_plot
    
    

def poly_fit(x, y, degree):
    x = np.array(x)  
    y = np.array(y)  
    
    # Reshape for sklearn
    X_features = x.reshape(-1, 1)
    y_h0 = y.reshape(-1, 1)
    
    # Build and train model
    model_h0 = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    model_h0.fit(X_features, y_h0)
    
    # Get the coefficients
    linear_model = model_h0.named_steps['linearregression']
    coefficients = linear_model.coef_[0]  # Get the coefficients for the polynomial features
    intercept = linear_model.intercept_[0]  # Get the intercept

    # Generate smooth x values for plotting the fitted curve
    x_plot = x.reshape(-1, 1)
    
    # Predict y values using the polynomial model
    y_pred = model_h0.predict(x_plot)
    
  

    print(f'Polynomial parameters: \n Coefficients : {coefficients} \n Intercept : {intercept}')

    return y_pred

def sig_fit(x,y):
    # Example input data (replace with your actual x and y)
    x = np.array(x)
    y = np.array(y)
    
    # Define the sigmoid function
    def sigmoid(x, L, k, x0):
        return L / (1 + np.exp(-k * (x - x0)))
    
    # Initial guess for parameters: L, k, x0
    initial_guess = [np.max(y), 1, np.median(x)]
    
    # Fit the sigmoid model
    params, _ = curve_fit(sigmoid, x, y, p0=initial_guess, maxfev=10000)
    
    # Extract fitted parameters
    L, k, x0 = params
    print(f"Fitted sigmoid parameters:\nL = {L:.4f}, k = {k:.4f}, x0 = {x0:.4f}")
    
    # Generate smooth x values for plotting
    x_plot = np.linspace(x.min(), x.max(), 100)
    
    # Predict y values using sigmoid model
    y_pred = sigmoid(x_plot, *params)

 

    return x_plot , y_pred
    
    
def histogram_plot(df, ax, alphas=1, colors="blue"):
    bin_edges = np.linspace(0, 1, 13)
    counts, _ = np.histogram(df, bins=len(bin_edges)-1)
    normalized_counts = counts / counts.sum()
    ax.bar(bin_edges[:-1], normalized_counts, width=np.diff(bin_edges), color=colors, edgecolor='black', align='edge', alpha=alphas)

    ax.set_xlabel('Mean_CAF')
    ax.set_ylabel('Normalized Count') 
    ax.set_xlim(bin_edges[0], bin_edges[-1])
    ax.grid(axis='y', alpha=0.5)


def Compute_tot_err(mod_dist,tot_err_list,obs_dist,tot_mce, tot_skwe, tot_stde, tot_arbe):
    mod_mean_caf = np.mean(mod_dist)
    mod_std = np.std(mod_dist)
    mod_skew = skew(mod_dist)
    
    err_mean_caf = abs(obs_mean_caf - mod_mean_caf)/abs(obs_mean_caf) 
    tot_mce.append(err_mean_caf)
    
    err_std = abs(obs_std - mod_std)/abs(obs_std)
    tot_stde.append(err_std)
    
    err_skew = abs(obs_skew - mod_skew)
    tot_skwe.append(err_skew)

    tot_error = np.sum([rel_err,err_mean_caf,err_std,err_skew])
    tot_err_list.append(tot_error)

# Saturation vapor mixing ratio at the ground

def compute_qsaturation(T, p):
    TTriple = 273.16  # Triple point of water in Kelvin
    epsWaterDryAir = 0.62198  # Ratio of molecular weights of water and dry air
        # Saturation vapour pressure of vapor using the Goff-Gratch formula
    es = 10**(10.79574 * (1 - TTriple / T) - 5.028 * np.log10(T / TTriple) +
              1.50475e-4 * (1 - 10**(-8.2969 * (T / TTriple - 1))) +
              0.42873e-3 * (10**(4.76955 * (1 - TTriple / T)) - 1) +
              0.78614 + 2)
    # Initial saturation specific humidity
    qs = epsWaterDryAir*es/p
    return qs
