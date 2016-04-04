# -*- coding: utf-8 -*-
#
# Author: Teemu Henrikki Pitkänen <teemu.pitkanen@helsinki.fi>
# University of Helsinki / Helsinki Institute for Information Technology 2016
#
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl

mpl.rcParams['grid.linestyle'] = "-"
mpl.rcParams['grid.color'] = (0, 0, 0)
#mpl.rcParams['grid.linewidth'] = 1#1.2
mpl.rcParams['grid.alpha'] = 0.5
#mpl.rcParams['axes.axisbelow'] = 1


# LOAD RESULTS
pqvresults = np.loadtxt('votingAndPqResults.txt')
with open('results.txt', 'r') as f:
    basicresults = np.array([[x for x in row.split() if x is not 'None'] for row in f.read().split('\n')][:-1])
    basicresults = np.array([x for x in basicresults if x[4] == 'None'])
    basicresults = np.concatenate((basicresults[:, :4], basicresults[:, 5:]), axis=1).astype(float)
votingresults = np.loadtxt('votingResults.txt').astype(float)


# PLOT THE RESULTS
fig, ax = plt.subplots(1,1,figsize=(10, 5))
ax.plot(pqvresults[:, -1], pqvresults[:, -2], '.', mec='#AAFFAA', mfc='#AAFFAA')
ax.plot(votingresults[:, -1], votingresults[:, -2], '.', mec='#FFAAAA', mfc='#FFAAAA')
ax.plot(basicresults[:, -1], basicresults[:, -2], '.', mec='#8888FF', mfc='#8888FF')
ax.legend(['Voting-MRPT with extra branches', 'Voting-MRPT', 'MRPT'], 'lower right')
ax.yaxis.set_ticks(np.arange(0,1.1,0.1))
ax.xaxis.set_ticks(np.arange(0,0.41,0.025))
ax.grid(True)
ax.axis((0, 0.4, 0, 1))
ax.set_xlabel('Query time (s)')
ax.set_ylabel('Accuracy in 10NN search (MNIST)')
#plt.savefig('mrptvsvoting.pdf', format='pdf', dpi=1000)

# FIND THE OPTIMAL PARAMETER CURVES
optimal_param_results_pq_v = []
optimal_param_results_voting = []
optimal_param_results_basic = []
for rtime in np.arange(0, 0.4, 0.005):
    pqvres = np.array([r for r in pqvresults if rtime < r[-1] < rtime + 0.005])
    vres = np.array([r for r in votingresults if rtime < r[-1] < rtime + 0.005])
    mrptres = np.array([r for r in basicresults if rtime < r[-1] < rtime + 0.005])

    if len(pqvres)>0:
        optimal_param_results_pq_v.append([pqvres[np.argmax(pqvres[:, -2]), -1], max(pqvres[:, -2])])
    if len(vres)>0:
        optimal_param_results_voting.append([vres[np.argmax(vres[:, -2]), -1], max(vres[:, -2])])
    if len(mrptres)>0:
        optimal_param_results_basic.append([mrptres[np.argmax(mrptres[:, -2]), -1], max(mrptres[:, -2])])

optimal_param_results_pq_v = np.array(optimal_param_results_pq_v).T
optimal_param_results_voting = np.array(optimal_param_results_voting).T
optimal_param_results_basic = np.array(optimal_param_results_basic).T

ax.plot(optimal_param_results_pq_v[0], optimal_param_results_pq_v[1], 'g-', lw=3)
ax.plot(optimal_param_results_voting[0], optimal_param_results_voting[1], 'r-', lw=3)
ax.plot(optimal_param_results_basic[0], optimal_param_results_basic[1], 'b-', lw=3)
print len(votingresults)

plt.show()