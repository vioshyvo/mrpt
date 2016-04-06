# -*- coding: utf-8 -*-
#
# Author: Teemu Henrikki Pitkänen <teemu.pitkanen@helsinki.fi>
# University of Helsinki / Helsinki Institute for Information Technology 2016
#
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl

# LOAD RESULTS
pqvresults = np.loadtxt('results_mnist/votingAndPqResults.txt')
basicresults = np.loadtxt('results_mnist/basicResults.txt')
votingresults = np.loadtxt('results_mnist/votingResults.txt')
rptresults = np.loadtxt('results_mnist/rptResults.txt')
pqvresults = pqvresults[np.argsort(pqvresults[:, -1])]
basicresults = basicresults[np.argsort(basicresults[:, -1])]
votingresults = votingresults[np.argsort(votingresults[:, -1])]
rptresults = rptresults[np.argsort(rptresults[:, -1])]


# PLOT THE RESULTS
mpl.rcParams['grid.linestyle'] = "-"
mpl.rcParams['grid.color'] = (0, 0, 0)
#mpl.rcParams['grid.linewidth'] = 1#1.2
mpl.rcParams['grid.alpha'] = 0.2
#mpl.rcParams['axes.axisbelow'] = 1
fig, ax = plt.subplots(1,1)
ax.plot(pqvresults[:, -1], pqvresults[:, -2], 'o', mec='k', mfc='g', alpha=0.1)
ax.plot(votingresults[:, -1], votingresults[:, -2], 'o', mec='k', mfc='r', alpha=0.1)
ax.plot(basicresults[:, -1], basicresults[:, -2], 'o', mec='k', mfc='b', alpha=0.1)
ax.plot(rptresults[:, -1], rptresults[:, -2], 'o', mec='k', mfc='k', alpha=0.1)
ax.legend(['Voting-MRPT with extra branches', 'Voting-MRPT', 'MRPT', 'Single-RPT'], 'lower right')
ax.yaxis.set_ticks(np.arange(0,1.1,0.1))
ax.xaxis.set_ticks(np.arange(0,0.41,0.05))
ax.grid(True)
ax.axis((0, 0.4, 0, 1))
ax.set_xlabel('Query time (s)')
ax.set_ylabel('Accuracy in 10NN search (MNIST)')
#plt.savefig('mrptvsvoting.pdf', format='pdf', dpi=1000)


# FIND THE OPTIMAL PARAMETER CURVES
def concavify(X):
    changes = 0
    while(1):
        i = 0
        while(1):
            if i >= len(X[0]) - 2:
                break
            if X[1, i] + (X[1, i+2]-X[1, i])*(X[0, i+1]-X[0, i])/(X[0, i+2]-X[0, i]) > X[1, i+1]:
                X = np.concatenate((X[:, :i+1], X[:, i+2:]), axis=1)
                changes += 1
            i += 1
        if changes == 0:
            break
        changes = 0
    #for i in range(1, len(X[0])):
    #    if X[1, i] < X[1, i-1]:
    #        X = X[:, :i]
    #        break
    return X

optimal_param_results_pq_v = concavify(pqvresults[:, [-1, -2]].T)[:,:-2]
optimal_param_results_voting = concavify(votingresults[:, [-1, -2]].T)
optimal_param_results_basic = concavify(basicresults[:, [-1, -2]].T)
optimal_param_results_rpt = concavify(rptresults[:, [-1, -2]].T)
h1, = ax.plot(optimal_param_results_pq_v[0], optimal_param_results_pq_v[1], 'g-', lw=3)
h2, = ax.plot(optimal_param_results_voting[0], optimal_param_results_voting[1], 'r-', lw=3)
h3, = ax.plot(optimal_param_results_basic[0], optimal_param_results_basic[1], 'b-', lw=3)
h4, = ax.plot(optimal_param_results_rpt[0], optimal_param_results_rpt[1], 'k-', lw=3)
ax.legend([h1, h2, h3, h4],['Voting-MRPT with extra branches', 'Voting-MRPT', 'MRPT', 'Single-RPT'], 'lower right')

plt.show()