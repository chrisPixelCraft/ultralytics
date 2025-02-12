import os, sys
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from sklearn.mixture import GaussianMixture as GMM

from scipy.stats import norm
import matplotlib as mpl
mpl.rcParams['axes.linewidth'] = 1.5
mpl.rcParams.update({'font.size': 15, 'font.family': 'STIXGeneral', 'mathtext.fontset': 'stix'})

def pulse(N, mean, var, select=2):
    window = [round(mean - select*var), round(mean + select*var)]

    pulse = []
    for i in range(N):
        if i >= window[0] and i <= window[1]:
            pulse.append(True)
        else:
            pulse.append(False)
    
    return np.asarray(pulse)*0.1

def travel_time_GMM():
    for reg_num in range(1, 12):
        travel_time = np.load(f'./travel_time/travel_time_{reg_num}.npy')

        data = []
        for sec, cnt in enumerate(travel_time):
            for _ in range(int(cnt)):
                data.append(sec)
        data = np.asarray(data).reshape(-1,1)

        travel_time = travel_time.reshape(-1,1)

        # confirm n_components through the value of bic (the smaller the better)
        min_bic = 200000000000000
        opt_bic = None
        for j in range(1,3): # 1 or 2
            gmm = GMM(n_components = j, max_iter=1000, random_state=0, covariance_type = 'spherical').fit(data)
            bic = gmm.bic(data)
            if bic < min_bic:
                min_bic = bic
                opt_bic = j

        # create GMM models and get the parameters
        gmm = GMM(n_components = opt_bic, max_iter=1000, random_state=0, covariance_type = 'spherical').fit(data)
        mean = gmm.means_  
        covs  = gmm.covariances_
        weights = gmm.weights_

        # create necessary things to plot
        x_axis = np.arange(travel_time.shape[0])
        plt.figure(figsize=(4,2))
        if opt_bic == 2:
            y_axis0 = norm.pdf(x_axis, float(mean[0]), np.sqrt(float(covs[0])))*weights[0] # 1st gaussian
            y_axis1 = norm.pdf(x_axis, float(mean[1]), np.sqrt(float(covs[1])))*weights[1] # 2nd gaussian

            density = travel_time[:, 0]/np.sum(travel_time[:, 0])

            window1 = pulse(travel_time.shape[0], float(mean[0]), np.sqrt(float(covs[0])))
            window2 = pulse(travel_time.shape[0], float(mean[1]), np.sqrt(float(covs[1])))

            plt.bar(x_axis, density, color='black', width=1.5)
            plt.plot(x_axis, y_axis0, lw=3, c='C0')
            plt.plot(x_axis, y_axis1, lw=3, c='C1')
            # plt.plot(x_axis, y_axis0+y_axis1, lw=3, c='C2', ls='dashed')
            plt.xlabel("Traveling Time (sec)", fontsize=16)
            plt.ylabel("Density", fontsize=16)

            plt.savefig(f'./travel_time/Reg{reg_num}.png', bbox_inches='tight')
            # plt.clf()

            plt.plot(x_axis, window1, color='blue')
            plt.plot(x_axis, window2, color='red')
            plt.savefig(f'./travel_time/Win{reg_num}.png', bbox_inches='tight')
            plt.clf()

        else:
            y_axis0 = norm.pdf(x_axis, float(mean[0]), np.sqrt(float(covs[0])))*weights[0]

            window1 = pulse(travel_time.shape[0], float(mean[0]), np.sqrt(float(covs[0])))

            density = travel_time[:, 0]/np.sum(travel_time[:, 0])
            plt.bar(x_axis, density, color='black', width=1.5)
            plt.plot(x_axis, y_axis0, lw=3, c='C0')
            plt.xlabel("X", fontsize=20)
            plt.ylabel("Density", fontsize=20)

            plt.savefig(f'./travel_time/Reg{reg_num}.png', bbox_inches='tight')

            plt.plot(x_axis, window1, color='blue')
            plt.savefig(f'./travel_time/Win{reg_num}.png', bbox_inches='tight')
            plt.clf()

if __name__ == "__main__":
    travel_time_GMM()
    