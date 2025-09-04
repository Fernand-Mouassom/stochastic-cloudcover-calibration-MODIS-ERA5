from utilities import linear_fit
from utilities import poly_fit
from utilities import sig_fit
from utilities import histogram_plot
from utilities import Compute_tot_err
from utilities import compute_qsaturation
import pickle
import os


# Manual Binning of data

#load dataset
df=pd.read_csv("grid_search_data.csv") 

#~~~~~~~~~~~~~~~~~~~~~~~~~
# Manual Binning of data
#~~~~~~~~~~~~~~~~~~~~~~~~~
num_bins_q = x  # set number of q bins
num_bins_T =  y # set number of T bins

df['q_bin'] = pd.cut(df['q'], bins=num_bins_q)
df['T_bin'] = pd.cut(df['T'], bins=num_bins_T)
bin_stats = df.groupby(['q_bin', 'T_bin']).agg(
    mean_CAF=('CAF', 'mean'),
    std_CAF=('CAF', 'std'),
    count=('CAF', 'count')
).reset_index()



# Define the base filename and the suffixes for the different scores
base_filename = 'score_distribution.pkl'
h_values = ['h05', 'h10', 'h12', 'h15', 'h17', 'h20', 'h22', 'h25', 'h30']

# Load data using a list comprehension
data1 = []
for h in h_values:
    filepath = f"{h}_{base_filename}"
    with open(filepath, 'rb') as f:
        data1.append(pickle.load(f))
        
        

# Initialize dictionaries to hold values for each h value
tot_values = {h: {'mce': [], 'skwe': [], 'stde': [], 'arbe': [], 'err': []} for h in h_values}

# Assuming num_distributions is the same for all
num_distributions = len(loaded_list2_h15)


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Computing total error for each moment
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Assuming you have the tot_values dictionary from earlier
for i in range(num_distributions):
    selected_T_bin = bin_stats['T_bin'].unique()[i]
    subset = df[df['T_bin'] == selected_T_bin]
    
    obs_mean_caf = np.mean(subset['CAF'])
    obs_std = np.std(subset['CAF'])
    obs_skew = skew(subset['CAF'])
    
    for h in h_values:
        Compute_tot_err(loaded_list2_h05[i], tot_values[h]['err'], subset['CAF'], 
                        tot_values[h]['mce'], tot_values[h]['skwe'], 
                        tot_values[h]['stde'], tot_values[h]['arbe'])

# Print results
for h in h_values:
    print(f"h={h} \n"
          f"total mean caf error = {np.mean(tot_values[h]['mce'])} \n"
          f"total skew error = {np.mean(tot_values[h]['skwe'])} \n"
          f"total standard deviation error = {np.mean(tot_values[h]['stde'])} \n"
          f"total relative bin error = {np.mean(tot_values[h]['arbe'])} \n"
          f"Overall total error = {np.mean(tot_values[h]['err'])}")

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Plotting components of total loss
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Calculate T_mean based on your bin_stats and df
T_mean = []
for i in range(len(bin_stats['T_bin'].unique())):
    selected_T_bin = bin_stats['T_bin'].unique()[i]
    subset = df[df['T_bin'] == selected_T_bin]
    T_mean.append(np.mean(subset["T"]))
T_mean = np.array(T_mean)

h_num = ["05", "10", "12", "15", "17", "20", "22", "25", "30"]
title = ["meancaf_error", "std_error", "skew_error"]
color = ['red', 'green', 'orange', 'purple', 'saddlebrown', 'pink', 'gray', 'cyan', 'magenta']
markers = ['o', 's', '^', 'D', 'v', 'P', '*', 'X', '<']  # Different marker styles

# Create subplots
fig, ax = plt.subplots(2, 2, figsize=(12, 7))
fig.tight_layout(pad=2.5) 
ax = ax.flatten()

for k in range(len(h_num)):
    df_h = data1[k]
    mse = []
    std = []
    skw = []
    
    for i in range(len(bin_stats['T_bin'].unique())):
        selected_T_bin = bin_stats['T_bin'].unique()[i]
        subset = df[df['T_bin'] == selected_T_bin]
        obs_mean_caf = subset['CAF'].mean()
        obs_std = subset['CAF'].std()
        obs_skew = skew(subset['CAF'])

        mod_subset = df_h[i]
        mod_mean_caf = np.array(mod_subset).mean()
        mod_std = np.array(mod_subset).std()
        mod_skew = skew(np.array(mod_subset))

        err_mean_caf = abs(obs_mean_caf - mod_mean_caf) / abs(obs_mean_caf)
        err_std = abs(obs_std - mod_std) / abs(obs_std)
        err_skew = abs(obs_skew - mod_skew) #/ abs(obs_skew)#################

        mse.append(err_mean_caf)
        std.append(err_std)
        skw.append(err_skew)

    mse = np.array(mse).flatten()
    std = np.array(std).flatten()
    skw = np.array(skw).flatten()

    # Plotting with T_mean as x-axis
    ax[0].plot(T_mean, mse, marker=markers[k % len(markers)], color=color[k], label=f"h_{h_num[k]}", linewidth=1.1, markersize=3.2)
    ax[1].plot(T_mean, std, marker=markers[k % len(markers)], color=color[k], label=f"h_{h_num[k]}", linewidth=1.1, markersize=3.2)
    ax[2].plot(T_mean, skw, marker=markers[k % len(markers)], color=color[k], label=f"h_{h_num[k]}", linewidth=1.1, markersize=3.2)

    ax[k % 3].set_xlim(min(T_mean), max(T_mean))
    ax[k % 3].set_xlabel("T_mean", fontsize=12, fontweight="bold")
    ax[k % 3].set_xlim(T_mean.min()-0.08, T_mean.max()+0.08)
    ax[k % 3].grid()


ax[0].set_ylabel("meancaf_error", fontsize=12, fontweight="bold")
ax[1].set_ylabel("std_error", fontsize=12, fontweight="bold")
ax[2].set_ylabel("skew_error", fontsize=12, fontweight="bold")

# Create a single legend outside of the plots
ax[1].legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='large', title="Legend")





markers = ['s', '*', 'o']

# Prepare lists for mean calculations
kk1 = []
kk2 = []
kk3 = []

# Loop through h_values to compute means
for h in h_values:
    kk1.append(np.mean(tot_values[h]['mce']))
    kk2.append(np.mean(tot_values[h]['skwe']))
    kk3.append(np.mean(tot_values[h]['stde']))





# Plot First Moment
ax[3].plot(h_values, kk1, marker=markers[0], color="#FF7F50", label='Avg Tot MCE')
ax[3].set_ylabel("Avg meancaf_error", fontsize=12, fontweight="bold", color="#FF7F50")
ax[3].tick_params(axis='y', labelcolor="#FF7F50")

# Create a second y-axis for the second moment
ax2 = ax[3].twinx()
ax2.plot(h_values, kk2, marker=markers[1], color="#4169E1", label='Avg Tot Skewness')
ax2.set_ylabel("Avg Skew_error", fontsize=12, fontweight="bold", color="#4169E1")
ax2.tick_params(axis='y', labelcolor="#4169E1")

# Create a third y-axis for the third moment
ax3 = ax[3].twinx()
ax3.plot(h_values, kk3, marker=markers[2], color="#32CD32", label='Avg Tot Std Dev')
ax3.spines['right'].set_position(('outward', 60))  # Offset the third y-axis
ax3.set_ylabel("Avg Std_error", fontsize=12, fontweight="bold", color="#32CD32")
ax3.tick_params(axis='y', labelcolor="#32CD32")

# Customize the plot
ax[3].set_xlabel("h Index", fontsize=12, fontweight="bold")
ax[3].grid()



plt.savefig("Figures/error_plots/sep_err_moment3.jpg", bbox_inches='tight', pad_inches=0.1, format='jpg', dpi=600)

plt.savefig('Figures/error_plots/sep_err_moment3.eps', bbox_inches='tight', pad_inches=0.1, format='eps', dpi=600)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Plotting histograms plots for bin distributions
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Bin distribution for h0 = 20
num_distributions = len(loaded_list2_h20)
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(10, 6)) 
axes = axes.flatten() 

for j,i in enumerate([0, 3, 5, 7, 9, 12]):
    
    selected_T_bin = bin_stats['T_bin'].unique()[i]
    subset = df[df['T_bin'] == selected_T_bin] 
    mean_obs=np.round(np.mean(subset['CAF']), 2)
    std_obs=np.round(np.std(subset['CAF']), 2)
    skw_obs=np.round(skew(subset['CAF']), 2)
    histogram_plot(subset['CAF'], ax=axes[j], alphas=1, colors="blue")

    #loaded_list2_h20[i] = loaded_list2_h20[i][:50]
    histogram_plot(loaded_list2_h20[i], ax=axes[j], alphas=0.7, colors="red")
    mean_model = np.round(np.mean(loaded_list2_h20[i]), 2)
    std_model = np.round(np.std(loaded_list2_h20[i]), 2)
    skw_model = np.round(skew(loaded_list2_h20[i]), 2)

    axes[j].text(0.48, 0.85, 'obs', ha='right', va='top', transform=axes[j].transAxes, fontsize=10, fontweight="bold", color="blue")
    axes[j].text(0.72, 0.85, 'mod', ha='right', va='top', transform=axes[j].transAxes, fontsize=10, fontweight="bold", color="red")

    axes[j].text(0.32, 0.95,"Temp:"   "\n\n\nMean:"  "\nSTD:"  "\nSkew:"  , 
                 ha='right', va='top', transform=axes[j].transAxes, fontsize=10, fontweight="bold", color="black")

    
    axes[j].text(0.50, 0.95, f'\n\n\n{mean_obs} \n{std_obs} \n{skw_obs}', 
                 ha='right', va='top', transform=axes[j].transAxes, fontsize=10, fontweight="bold", color="blue")

    axes[j].text(0.60, 0.95, f'\n\n\n{mean_model} \n{std_model} \n{skw_model}' , 
                 ha='left', va='top', transform=axes[j].transAxes, fontsize=10, fontweight="bold", color="red")

    axes[j].text(0.45, 0.95, f'{np.mean(np.round(subset['T'].mean(),2))}' , 
                 ha='left', va='top', transform=axes[j].transAxes, fontsize=10, fontweight="bold", color="black")

plt.tight_layout()

plt.savefig("Figures/error_plots/h0_20score.jpg", bbox_inches='tight', pad_inches=0.1, format='jpg', dpi=600)
plt.savefig("Figures/error_plots/h0_20score.eps", bbox_inches='tight', pad_inches=0.1, format='eps', dpi=600)


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Plotting Optimal J0 values and fittiing with polynomial and sigmoid fit
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Initialize a list for the scores
Jh_scores = []

# Loop through h values to construct file names and load data
for h in h_values:
    filename = f'J{h}_score_distribution.pkl'
    with open(filename, 'rb') as f:
        Jh_scores.append(pickle.load(f))

# Alternatively, you can keep the same variable names if needed
Jh05_score, Jh10_score, Jh12_score, Jh15_score, Jh17_score, Jh20_score, Jh22_score, Jh25_score, Jh30_score = Jh_scores

T_mean = []
for i in range(len(bin_stats['T_bin'].unique())):
    selected_T_bin = bin_stats['T_bin'].unique()[i]
    subset = df[df['T_bin'] == selected_T_bin]
    T_mean.append(np.mean(subset["T"]))
T_mean = np.array(T_mean)

markers = ['s','*','o']


fig, ax = plt.subplots(2, 3, figsize=(15, 8))
fig.tight_layout(pad=3.0)

# First Moment
ax[0, 0].scatter(T_mean, Jh20_mse, marker=markers[0] ,color="blue", label='1 moment loss')
ax[0, 0].legend()
ax[0, 0].set_xlabel("T_bin_mean", fontsize=12, fontweight="bold")
ax[0, 0].set_ylabel("J0", fontsize=12, fontweight="bold")
ax[0, 0].set_title("First Moment", fontsize=12, fontweight="bold")
ax[0, 0].grid()

# Second Moment 
ax[0, 1].scatter(T_mean, Jh20_msestd, marker=markers[1], color="red", label='2 moments loss')
ax[0, 1].legend()
ax[0, 1].set_xlabel("T_bin_mean", fontsize=12, fontweight="bold")
ax[0, 1].set_ylabel("J0", fontsize=12, fontweight="bold")
ax[0, 1].set_title("Second Moment", fontsize=12, fontweight="bold")
ax[0, 1].grid()

# Third Moment 
ax[0, 2].scatter(T_mean, Jh20_score, marker=markers[2], color="orange", label='3 moment loss')
ax[0, 2].legend()
ax[0, 2].set_xlabel("T_bin_mean", fontsize=12, fontweight="bold")
ax[0, 2].set_ylabel("J0", fontsize=12, fontweight="bold")
ax[0, 2].set_title("Third Moment", fontsize=12, fontweight="bold")
ax[0, 2].grid()

# Combined plots
ax[1, 0].scatter(T_mean, Jh20_mse, marker=markers[0], color="blue", label='1 moment loss')
ax[1, 0].scatter(T_mean, Jh20_msestd, marker=markers[1], color="red", label='2 moment loss')
ax[1, 0].scatter(T_mean, Jh20_score, marker=markers[2], color="orange", label='3 moment loss')
ax[1, 0].legend()
ax[1, 0].set_xlabel("T_bin_mean", fontsize=12, fontweight="bold")
ax[1, 0].set_ylabel("J0", fontsize=12, fontweight="bold")
ax[1, 0].set_title("combined plot", fontsize=12, fontweight="bold")
ax[1, 0].grid()

# first fit plots
ax[1, 1].scatter(T_mean, Jh20_score, marker=markers[2], color="orange", label='mse_std')
ax[1, 1].plot(sig_fit(T_mean, Jh20_score)[0], sig_fit(T_mean, Jh20_score)[1] , color='red', linewidth=2, label=f'Sigmoid Fit') #Sigmoid fit
ax[1, 1].legend()
ax[1, 1].set_xlabel("T_bin_mean", fontsize=12, fontweight="bold")
ax[1, 1].set_ylabel("J0", fontsize=12, fontweight="bold")
ax[1, 1].set_title("Sigmoid Regression Fit", fontsize=12, fontweight="bold")
ax[1, 1].grid()

# second fit plots
ax[1, 2].scatter(T_mean, Jh20_score, marker=markers[2], color="orange", label='mse_std')
ax[1, 2].plot(T_mean, poly_fit(T_mean, Jh20_score, degree =3), color='red', linewidth=2, label=f'Polynomial Fit deg 3') #polynomial fit
ax[1, 2].legend()
ax[1, 2].set_xlabel("T_bin_mean", fontsize=12, fontweight="bold")
ax[1, 2].set_ylabel("J0", fontsize=12, fontweight="bold")
ax[1, 2].set_title("Polynomial Regression Fit", fontsize=12, fontweight="bold")
ax[1, 2].grid()


plt.savefig("Figures/error_plots/j0_values_h20.jpg", bbox_inches='tight', pad_inches=0.1, format='jpg', dpi=600)
plt.savefig("Figures/error_plots/j0_values_h20.eps", bbox_inches='tight', pad_inches=0.1, format='eps', dpi=600)


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Testing of the Universal Power Law between j0 & h0
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

gg = [Jh05_score,Jh10_score,Jh12_score,Jh15_score,Jh17_score,Jh20_score,Jh22_score,Jh25_score,Jh30_score]
h0 = [5, 10, 12, 15, 17, 20, 22, 25, 30]

j0_mean = []
for i in range(len(gg)):
    j0_mean.append(np.mean(gg[i]))

y = np.array(j0_mean)
x = np.array(h0)

log_x = np.log(x)
log_y = np.log(y)

y_fit, x_fit = linear_fit(log_x, log_y) # Linearly fitting the data to find alpha


#plotting the results
fig, ax = plt.subplots(1, 2, figsize=(14, 6))
fig.tight_layout(pad=3.0) 

# First subplot for the first fit plot
ax[0].scatter(log_y, log_x, color="black", label='$h_0$')
ax[0].plot(log_y, log_x, '--', color='red', label='Linear fit')
ax[0].set_xlabel("$\log ~~h_0$", fontsize=12, fontweight="bold")
ax[0].set_ylabel("$\log ~~J_0$", fontsize=12, fontweight="bold")
ax[0].grid()
ax[0].legend()

# Second subplot for the second fit plot
alpha = 0.90134779
L = []
for i in range(2, len(gg) - 1):
    l = np.array(gg[i]) / (h0[i] ** alpha)
    L.append(l)

    ax[1].scatter(T_mean, l, color=color[i], label=f'sigmoid fit, $h_0$: {h0[i]}')
    y, x = sig_fit(T_mean, l)
    ax[1].plot(y, x, color=color[i])

ax[1].set_xlabel("Mean_T per bin", fontsize=12,fontweight="bold")
ax[1].set_ylabel(r"$J_0~ /~h_0^\alpha$", fontsize=12, fontweight="bold")
ax[1].grid()
ax[1].legend()

plt.savefig("Figures/error_plots/powerlaw_j0_h0.jpg", bbox_inches='tight', pad_inches=0.1, format='jpg', dpi=600)
plt.savefig("Figures/error_plots/powerlaw_j0_h0.eps", bbox_inches='tight', pad_inches=0.1, format='eps', dpi=600)


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Testing for the Universal Power Law for CAF
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

T_mean = []
mean_CAF = []
for i in range(len(bin_stats['T_bin'].unique())):
    selected_T_bin = bin_stats['T_bin'].unique()[i]
    subset = df[df['T_bin'] == selected_T_bin]
    T_mean.append(np.mean(subset["T"]))
    mean_CAF.append(np.mean(subset["CAF"]))

T_mean = np.array(T_mean)
mean_CAF = np.array(mean_CAF)

log_T_mean = np.log(T_mean)
log_mean_CAF = np.log(mean_CAF)

y_fit, x_fit = linear_fit(log_T_mean, log_mean_CAF)

# Your objects list
h_num = ["c) Observation CAF vs T", "d) Model CAF vs T"]
objects = [loaded_list2_h05,loaded_list2_h20]

# Create a figure with 2 rows and 4 columns
fig, ax = plt.subplots(1, 2, figsize=(14, 6))

# Adjust the spacing between subplots
plt.subplots_adjust(wspace=0.3, hspace=0.4)  # Increase wspace for horizontal spacing

# Iterate over the objects and plot each one in a subplot
for i, obj in enumerate(objects):

    # Prepare T_mean and mean_CAF
    T_mean = []
    mean_CAF = []

    if i == 0:
        for j in range(len(bin_stats['T_bin'].unique())):
            selected_T_bin = bin_stats['T_bin'].unique()[j]
            subset = df[df['T_bin'] == selected_T_bin]
            T_mean.append(np.mean(subset["T"]))
            mean_CAF.append(np.mean(subset["CAF"])) 

    else:                                                           
            
        for j in range(len(bin_stats['T_bin'].unique())):
            selected_T_bin = bin_stats['T_bin'].unique()[j]
            subset = df[df['T_bin'] == selected_T_bin]
            T_mean.append(np.mean(subset["T"]))
        
            obs_distr = obj[j]  # Use objects list
            mean_CAF.append(np.mean(obs_distr))
    
    T_mean = np.array(T_mean)
    mean_CAF = np.array(mean_CAF)
    
    log_T_mean = np.log(T_mean)
    log_mean_CAF = np.log(mean_CAF)
    
    print('\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print(f'h: {h_num[i]}')
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

    y_fit, x_fit = linear_fit(log_T_mean, log_mean_CAF)

    # Determine subplot position
    row = i 
    col = i 

    # Plot the data
    ax[row].scatter( log_T_mean, log_mean_CAF, color="black")
    ax[row].plot(x_fit,  y_fit,  '--', color='red', label='Linear fit')
    ax[row].set_xlabel("log T_mean", fontsize=12,fontweight="bold")
    ax[row].set_ylabel("log Mean CAF", fontsize=12, fontweight="bold")
    ax[row].grid()
    ax[row].legend()
    ax[row].set_title(f'{h_num[i]}', fontsize=14, fontweight="bold")  # Title for each subplot


plt.savefig("Figures/error_plots/powerlaw_CAF_T.jpg", bbox_inches='tight', pad_inches=0.1,  format='jpg', dpi=600)
plt.savefig("Figures/error_plots/powerlaw_CAF_T.eps", bbox_inches='tight', pad_inches=0.1, format='eps', dpi=600)
