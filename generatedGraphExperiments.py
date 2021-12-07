import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import numpy
import networkx as nx
import math
import graphfunctions as gf #import the needed graph functions
from tqdm import tqdm
import collections

import threading
import matplotlib.pyplot as plt
from numpy.random import choice
import random
import time
from pathlib import Path
import os

#Probabilistic Edge Removal

def opinionize(new_opinion, index):
    return new_opinion[index]

def remove_edges(G, new_opinion, n):
    #Forming a distribution of edges and removing n of them
    edges_list = np.array(G.edges)
    n_edges = len(edges_list)
    
    opinion_list = opinionize(new_opinion, edges_list)
    
    minus = np.array([1, -1])
    disr = np.dot(opinion_list, minus)
    disr = np.abs(disr)  #get disagreement in a numpy way

    disr = disr / sum(disr)

    to_remove = choice(np.arange(n_edges), n, replace=False, p=disr)
    to_remove_edges = edges_list[to_remove]

    counter = 0
    for u,v in to_remove_edges:
        if G.degree[u] != 1 and G.degree[v] != 1:
            G.remove_edge(u, v)
            counter += 1

    return counter

def remove_RANDOM_edges(G, new_opinion, n):
    #Forming a distribution of edges and removing n of them
    edges_list = np.array(G.edges)
    n_edges = len(edges_list)

    to_remove = choice(np.arange(n_edges), n, replace=False)
    to_remove_edges = edges_list[to_remove]

    counter = 0
    for u,v in to_remove_edges:
        if G.degree[u] != 1 and G.degree[v] != 1:
            G.remove_edge(u, v)
            counter += 1

    return counter

def remove_edges_not_in_list(G, new_opinion, n, off_limits_set):
    #Forming a distribution of edges and removig n of them
    edges_list = np.array(G.edges)
        
    #take edges that are in the off limits list out of consideration
    off_limits_indices = []
    for index, edge in enumerate(edges_list):
        if ((edge[0], edge[1]) in off_limits_set or (edge[1], edge[0]) in off_limits_set):
            off_limits_indices.append(index)

    edges_list = np.delete(edges_list, off_limits_indices, 0)
    n_edges = len(edges_list)

    opinion_list = opinionize(new_opinion, edges_list)
    minus = np.array([1, -1])
    disr = np.dot(opinion_list, minus)
    disr = np.abs(disr)  #get disagreement in a numpy way

    disr = disr / sum(disr)

    to_remove = choice(np.arange(n_edges), n, replace=False, p=(disr))
    to_remove_edges = edges_list[to_remove]

    counter = 0
    for u,v in to_remove_edges:
        if G.degree[u] != 1 and G.degree[v] != 1:
            G.remove_edge(u, v)
            counter += 1

    return counter


# CODE FOR BIMODAL INNATE OPINION (can replace [random.uniform(-1,1) for _ in range(n)])
    # innate_opinion = [np.random.normal(-0.7,0.2) for _ in range(int(n/2))]
    # innate_opinion2 = [np.random.normal(0.7,0.2) for _ in range(int(n/2))]
    # innate_opinion.extend(innate_opinion2)


def exp1(numDegree, rootFolder):
    # Experiment Number 1:
    # Modulating the size of random graphs (number of nodes)

    # Setup:
    # Opinion Initial : [-1 to 1]
    # Keeping the average degree constant at "numDegree"
    # FOF recommendation and confirmation bias edge deletion
    # Plot polarization, disagreement

    nodes = [250, 500, 1000, 2500, 5000, 10000]
    polarizations = collections.defaultdict(list)
    disagreements = collections.defaultdict(list)
    snapshots = collections.defaultdict(list)
    opinions = collections.defaultdict(list)

    similarities = collections.defaultdict(list)

    iterations = 500
    percent = 0.1 # remove 10% of edges at each step
    degree = numDegree
    trials = 5

    for o,n in tqdm(enumerate(nodes)):
        p = degree/n
        identity = np.identity(n, dtype = float)
        for q in tqdm(range(trials)):
            G = nx.generators.random_graphs.gnp_random_graph(n,p)
            innate_opinion = [random.uniform(-1,1) for _ in range(n)]
            laplace = nx.linalg.laplacian_matrix(G)

            polarization = []
            disagreement = []
            G_over_time = []
            changed_edges = []
            opinions_over_time = []

            new_opinion = gf.change_fj_opinions(laplace, innate_opinion, identity)
            polarization.append(gf.calculate_polarization(new_opinion))
            disagreement.append(gf.calculate_disagreement(laplace, new_opinion))
            G_over_time.append(G.copy())
            opinions_over_time.append(innate_opinion)

            for i in tqdm(range(iterations)):
                r = remove_edges(G, new_opinion, int(G.number_of_edges() * percent) )
                gf.add_new_friend_for_each_node(G,r,n)
                laplace = nx.linalg.laplacian_matrix(G)
                new_opinion = gf.change_fj_opinions(laplace, innate_opinion, identity)

                if (i % 20 == 0):
                    G_over_time.append(G.copy())
                    opinions_over_time.append(new_opinion)
                polarization.append(gf.calculate_polarization(new_opinion))
                disagreement.append(gf.calculate_disagreement(laplace, new_opinion))

            polarizations[o].append(polarization)
            disagreements[o].append(disagreement)
            snapshots[o].append(G_over_time)
            opinions[o].append(opinions_over_time)
            similarities[o].append(changed_edges)

    if not os.path.isdir(rootFolder):
        os.makedirs(rootFolder)
    if not os.path.isdir(rootFolder + "histograms/"):
        os.makedirs(rootFolder + "histograms/")

    for iter, node in enumerate(nodes):
        fig_folder = Path(rootFolder + "histograms/" + str(node))
        if not os.path.isdir(fig_folder):
            os.makedirs(fig_folder)
        for i in range(26):
            for j in range(trials):
                plt.hist(np.array(opinions[iter][j][i]), bins = numpy.arange(-1, 1, 0.05))
            plt.title("Histogram for graph size " + str(node) + " @ " + str(i*20) + " iterations")
            plt.savefig(fig_folder / str(i*20), facecolor='w', transparent=False)
            plt.clf()

    fig_folder = Path(rootFolder)

    final_polarizations = []
    final_disagreements = []
    length = len(nodes)
    for i in range(length):
        final_polarizations.append(np.mean(np.array(polarizations[i]), axis = 0))
        final_disagreements.append(np.mean(np.array(disagreements[i]), axis = 0))

    iterations_array = numpy.arange(iterations + 1)

    print("Polarization plot over time")
    for i in range(length):
        plt.plot(iterations_array, np.array(final_polarizations[i]))
    plt.legend(nodes)
    plt.ylabel('Polarization')
    plt.savefig(fig_folder / "polarization_over_time", facecolor='w', transparent=False)
    plt.clf()

    print("Plotting polarization versus nodes")
    final_p = []
    for i in range(length):
      final_p.append(np.array(final_polarizations[i])[-1])
    plt.plot(nodes, final_p, marker='o')
    plt.ylabel('Polarization')
    plt.savefig(fig_folder / "polarization_vs_nodes", facecolor='w', transparent=False)
    plt.clf()

    print("Disagreement plot over time")
    for i in range(length):
        plt.plot(iterations_array, np.array(final_disagreements[i]))
    plt.legend(nodes)
    plt.ylabel('Disagreement')
    plt.savefig(fig_folder / "disagreement_over_time", facecolor='w', transparent=False)
    plt.clf()

    print("Polarization&Disagreement plot over time")
    for i in range(length):
        plt.plot(iterations_array, np.array(final_polarizations[i])+np.array(final_disagreements[i]))
    plt.legend(nodes)
    plt.ylabel('PD(L)')
    plt.savefig(fig_folder / "PDL_over_time", facecolor='w', transparent=False)
    plt.clf()

    print("Plotting Disagreement versus nodes")
    final_d = []
    for i in range(length):
      final_d.append(np.array(final_disagreements[i])[-1])
    plt.plot(nodes, final_d, marker='o')
    plt.ylabel('Disagreement')
    plt.savefig(fig_folder / "disagreement_vs_nodes", facecolor='w', transparent=False)
    plt.clf()

def exp3(rootFolder):
    # Experiment Number 3:
    # Varying connection probabilities, showing that very dense graphs are resistant to polarization

    # Setup:
    # Opinion Initial : [-1 to 1]
    # FOF recommendation and confirmation bias edge deletion
    # Plot polarization, disagreement

    n = 1000
    percent = 0.1 # remove 10% of edges at each step
    iterations = 5000
    identity = np.identity(n, dtype = float)

    connection_probs = [0.025, 0.04, 0.05, 0.06, 0.075, 0.1]
    trials = 5
    polarizations = collections.defaultdict(list)
    disagreements = collections.defaultdict(list)
    snapshots = collections.defaultdict(list)
    opinions = collections.defaultdict(list)

    for o,p in tqdm(enumerate(connection_probs)):
        for q in tqdm(range(trials)):
            innate_opinion = [random.uniform(-1,1) for _ in range(n)]
            G = nx.generators.random_graphs.gnp_random_graph(n,p)
            A = nx.linalg.graphmatrix.adjacency_matrix(G)
            laplace = nx.linalg.laplacian_matrix(G)

            polarization = []
            disagreement = []
            G_over_time = []
            opinions_over_time = []

            # Apply FJ opinion model to get the first equilibrium state
            new_opinion = gf.change_fj_opinions(laplace, innate_opinion, identity)
            polarization.append(gf.calculate_polarization(new_opinion))
            disagreement.append(gf.calculate_disagreement(laplace, new_opinion))
            G_over_time.append(G.copy())
            opinions_over_time.append(innate_opinion)

            for i in tqdm(range(iterations)):
                r = remove_edges(G, new_opinion, int(G.number_of_edges() * percent) )
                gf.add_new_friend_for_each_node(G,r,n)
                laplace = nx.linalg.laplacian_matrix(G)
                new_opinion = gf.change_fj_opinions(laplace, innate_opinion, identity)

                if (i % 20 == 0):
                    G_over_time.append(G.copy())
                    opinions_over_time.append(new_opinion)
                polarization.append(gf.calculate_polarization(new_opinion))
                disagreement.append(gf.calculate_disagreement(laplace, new_opinion))
            polarizations[o].append(polarization)
            disagreements[o].append(disagreement)
            snapshots[o].append(G_over_time)
            opinions[o].append(opinions_over_time)

    if not os.path.isdir(rootFolder):
        os.makedirs(rootFolder)
    if not os.path.isdir(rootFolder + "histograms/"):
        os.makedirs(rootFolder + "histograms/")

    for iter, probs in enumerate(connection_probs):
        fig_folder = Path(rootFolder + "histograms/" + str(probs))
        if not os.path.isdir(fig_folder):
            os.makedirs(fig_folder)
        for i in range(26):
            for j in range(trials):
                plt.hist(np.array(opinions[iter][j][i]), bins = numpy.arange(-1, 1, 0.05))
            plt.title("Histogram for connection probability " + str(probs) + " @ " + str(i*20) + " iterations")
            plt.savefig(fig_folder / str(i*20), facecolor='w', transparent=False)
            plt.clf()

    final_polarizations = []
    final_disagreements = []
    length = len(connection_probs)
    for i in range(length):
        final_polarizations.append(numpy.mean(numpy.array(polarizations[i]), axis = 0))
        final_disagreements.append(numpy.mean(numpy.array(disagreements[i]), axis = 0))

    iterations_array = numpy.arange(iterations + 1)

    fig_folder = Path(rootFolder)

    print("Polarization plot over time")
    for i in range(length):
        plt.plot(iterations_array, np.array(final_polarizations[i]))
    plt.legend(connection_probs)
    plt.ylabel('Polarization')
    plt.savefig(fig_folder / "polarization_over_time", facecolor='w', transparent=False)
    plt.clf()

    print("Plotting polarization versus Connection probs")
    final_p = []
    for i in range(length):
      final_p.append(np.array(final_polarizations[i])[-1])
    plt.plot(connection_probs, final_p, marker='o')
    plt.ylabel('Polarization')
    plt.savefig(fig_folder / "polarization_vs_probs", facecolor='w', transparent=False)
    plt.clf()

    print("Disagreement plot over time")
    for i in range(length):
        plt.plot(iterations_array, np.array(final_disagreements[i]))
    plt.legend(connection_probs)
    plt.ylabel('Disagreement')
    plt.savefig(fig_folder / "disagreement_over_time", facecolor='w', transparent=False)
    plt.clf()

    print("Polarization&Disagreement plot over time")
    for i in range(length):
        plt.plot(iterations_array, np.array(final_polarizations[i])+np.array(final_disagreements[i]))
    plt.legend(connection_probs)
    plt.ylabel('PD(L)')
    plt.savefig(fig_folder / "PDL_over_time", facecolor='w', transparent=False)
    plt.clf()

    print("Plotting disagreement versus Connection probs")
    final_d = []
    for i in range(length):
      final_d.append(np.array(final_disagreements[i])[-1])
    plt.plot(connection_probs, final_d, marker='o')
    plt.ylabel('Disagreement')
    plt.savefig(fig_folder / "disagreement_vs_connprob", facecolor='w', transparent=False)
    plt.clf()


def exp4(rootFolder):
    # Experiment Number 4:
    # Varying the percentage of edges removed/added at each step

    # Setup:
    # Opinion Initial : [-1 to 1]
    # FOF recommendation and confirmation bias edge deletion
    # Plot polarization, disagreement

    n = 5000
    iterations = 500

    p = 0.01
    removal_edges = [0.01, 0.05, 0.1, 0.15, 0.2]
    trials = 5
    polarizations = collections.defaultdict(list)
    disagreements = collections.defaultdict(list)
    snapshots = collections.defaultdict(list)
    opinions = collections.defaultdict(list)
    identity = np.identity(n, dtype = float)

    for o,rem in tqdm(enumerate(removal_edges)):
        for q in tqdm(range(trials)):
            innate_opinion = [random.uniform(-1,1) for _ in range(n)]
            G = nx.generators.random_graphs.gnp_random_graph(n,p)
            A = nx.linalg.graphmatrix.adjacency_matrix(G)
            laplace = nx.linalg.laplacian_matrix(G)

            polarization = []
            disagreement = []
            G_over_time = []
            opinions_over_time = []

            # Apply FJ opinion model to get the first equilibrium state
            new_opinion = gf.change_fj_opinions(laplace, innate_opinion, identity)
            polarization.append(gf.calculate_polarization(new_opinion))
            disagreement.append(gf.calculate_disagreement(laplace, new_opinion))
            G_over_time.append(G.copy())
            opinions_over_time.append(innate_opinion)
            for i in tqdm(range(iterations)):
                r = remove_edges(G, new_opinion, int(G.number_of_edges() * rem) )
                gf.add_new_friend_for_each_node(G,r,n)
                laplace = nx.linalg.laplacian_matrix(G)
                new_opinion = gf.change_fj_opinions(laplace, innate_opinion, identity)

                if (i % 20 == 0):
                    G_over_time.append(G.copy())
                    opinions_over_time.append(new_opinion)
                polarization.append(gf.calculate_polarization(new_opinion))
                disagreement.append(gf.calculate_disagreement(laplace, new_opinion))
            polarizations[o].append(polarization)
            disagreements[o].append(disagreement)
            snapshots[o].append(G_over_time)
            opinions[o].append(opinions_over_time)

    if not os.path.isdir(rootFolder):
        os.makedirs(rootFolder)
    if not os.path.isdir(rootFolder + "histograms/"):
        os.makedirs(rootFolder + "histograms/")

    for iter, rem in enumerate(removal_edges):
        fig_folder = Path(rootFolder + "histograms/" + str(rem))
        if not os.path.isdir(fig_folder):
            os.makedirs(fig_folder)
        for i in range(26):
            for j in range(trials):
                plt.hist(np.array(opinions[iter][j][i]), bins = numpy.arange(-1, 1, 0.05))
            plt.title("Histogram for removing edge percentage " + str(rem) + " @ " + str(i*20) + " iterations")
            plt.savefig(fig_folder / str(i*20), facecolor='w', transparent=False)
            plt.clf()

    final_polarizations = []
    final_disagreements = []
    length = len(removal_edges)
    for i in range(length):
        final_polarizations.append(np.mean(np.array(polarizations[i]), axis = 0))
        final_disagreements.append(np.mean(np.array(disagreements[i]), axis = 0))

    iterations_array = numpy.arange(iterations + 1)

    fig_folder = Path(rootFolder)

    print("Polarization plot over time")
    for i in range(length):
        plt.plot(iterations_array, np.array(final_polarizations[i]))
    plt.legend(removal_edges)
    plt.ylabel('Polarization')
    plt.savefig(fig_folder / "polarization_over_time", facecolor='w', transparent=False)
    plt.clf()

    print("Plotting polarization versus Edge Removal")
    final_p = []
    for i in range(length):
      final_p.append(np.array(final_polarizations[i])[-1])
    plt.plot(removal_edges, final_p, marker='o')
    plt.ylabel('Polarization')
    plt.savefig(fig_folder / "polarization_vs_edges", facecolor='w', transparent=False)
    plt.clf()

    print("Disagreement plot over time")
    for i in range(length):
        plt.plot(iterations_array, np.array(final_disagreements[i]))
    plt.legend(removal_edges)
    plt.ylabel('Disagreement')
    plt.savefig(fig_folder / "disagreement_over_time", facecolor='w', transparent=False)
    plt.clf()

    print("Polarization&Disagreement plot over time")
    for i in range(length):
        plt.plot(iterations_array, np.array(final_polarizations[i])+np.array(final_disagreements[i]))
    plt.legend(removal_edges)
    plt.ylabel('PD(L)')
    plt.savefig(fig_folder / "PDL_over_time", facecolor='w', transparent=False)
    plt.clf()

    print("Plotting disagreement versus Edge Removal")
    final_d = []
    for i in range(length):
      final_d.append(np.array(final_disagreements[i])[-1])
    plt.plot(removal_edges, final_d, marker='o')
    plt.ylabel('Disagreement')
    plt.savefig(fig_folder / "disagreement_vs_edges", facecolor='w', transparent=False)
    plt.clf()


def exp6(rootFolder):
    # Experiment Number 6:
    # Showing that fixed edges have an effect on polarization

    # Setup:
    # Opinion Initial : [-1 to 1]
    # FOF recommendation and confirmation bias edge deletion
    # Fixed edges in the graph
    # Plot polarization, disagreement

    n = 1000
    set_percent = [0.0, 0.01, 0.025, 0.05, 0.075, 0.1, 0.15]
    polarizations = collections.defaultdict(list)
    disagreements = collections.defaultdict(list)
    snapshots = collections.defaultdict(list)
    opinions = collections.defaultdict(list)

    offLimitsLists = collections.defaultdict(list)

    iterations = 500
    percent = 0.1 # remove 10% of edges at each step
    degree = 40
    trials = 5
    for o,per in tqdm(enumerate(set_percent)):
        p = degree/n
        identity = np.identity(n, dtype = float)
        for q in tqdm(range(trials)):
            G = nx.generators.random_graphs.barabasi_albert_graph(n,10)
            innate_opinion = [random.uniform(-1,1) for _ in range(n)]
            laplace = nx.linalg.laplacian_matrix(G)

            # Generate a percentage of edges in the graph to be fixed, store in "off_limits_set"
            num_edges = len(G.edges)
            oL = np.array(G.edges)[choice(num_edges, int(num_edges*per), replace=False),:]
            off_limits_set = set()
            for edge in oL:
                off_limits_set.add((edge[0], edge[1]))
            offLimitsLists[o].append(off_limits_set)
                    
            polarization = []
            disagreement = []
            G_over_time = []
            changed_edges = []
            opinions_over_time = []

            new_opinion = gf.change_fj_opinions(laplace, innate_opinion, identity)
            polarization.append(gf.calculate_polarization(new_opinion))
            disagreement.append(gf.calculate_disagreement(laplace, new_opinion))
            G_over_time.append(G.copy())
            opinions_over_time.append(innate_opinion)

            for i in tqdm(range(iterations)):
                r = remove_edges_not_in_list(G, new_opinion, int(G.number_of_edges() * percent), off_limits_set)
                gf.add_new_friend_for_each_node(G,r,n)
                laplace = nx.linalg.laplacian_matrix(G)
                new_opinion = gf.change_fj_opinions(laplace, innate_opinion, identity)

                if (i % 20 == 0):
                    G_over_time.append(G.copy())
                    opinions_over_time.append(new_opinion)
                polarization.append(gf.calculate_polarization(new_opinion))
                disagreement.append(gf.calculate_disagreement(laplace, new_opinion))

            polarizations[o].append(polarization)
            disagreements[o].append(disagreement)
            snapshots[o].append(G_over_time)
            opinions[o].append(opinions_over_time)

    if not os.path.isdir(rootFolder):
        os.makedirs(rootFolder)
    if not os.path.isdir(rootFolder + "histograms/"):
        os.makedirs(rootFolder + "histograms/")

    for iter, per in enumerate(set_percent):
        fig_folder = Path(rootFolder + "histograms/" + str(per))
        if not os.path.isdir(fig_folder):
            os.makedirs(fig_folder)
        for i in range(26):
            for j in range(trials):
                plt.hist(np.array(opinions[iter][j][i]), bins = numpy.arange(-1, 1, 0.05))
            plt.title("Histogram for fixed edges percent " + str(per) + " @ " + str(i*20) + " iterations")
            plt.savefig(fig_folder / str(i*20), facecolor='w', transparent=False)
            plt.clf()

    fig_folder = Path(rootFolder)

    final_polarizations = []
    final_disagreements = []
    length = len(set_percent)
    for i in range(length):
        final_polarizations.append(np.mean(np.array(polarizations[i]), axis = 0))
        final_disagreements.append(np.mean(np.array(disagreements[i]), axis = 0))

    iterations_array = numpy.arange(iterations + 1)

    print("Polarization plot over time")
    for i in range(length):
        plt.plot(iterations_array, np.array(final_polarizations[i]))
    plt.legend(set_percent)
    plt.ylabel('Polarization')
    plt.savefig(fig_folder / "polarization_over_time", facecolor='w', transparent=False)
    plt.clf()

    print("Plotting polarization versus fixed percent")
    final_p = []
    for i in range(length):
      final_p.append(np.array(final_polarizations[i])[-1])
    plt.plot(set_percent, final_p, marker='o')
    plt.ylabel('Polarization')
    plt.savefig(fig_folder / "polarization_vs_nodes", facecolor='w', transparent=False)
    plt.clf()

    print("Disagreement plot over time")
    for i in range(length):
        plt.plot(iterations_array, np.array(final_disagreements[i]))
    plt.legend(set_percent)
    plt.ylabel('Disagreement')
    plt.savefig(fig_folder / "disagreement_over_time", facecolor='w', transparent=False)
    plt.clf()

    print("Polarization&Disagreement plot over time")
    for i in range(length):
        plt.plot(iterations_array, np.array(final_polarizations[i])+np.array(final_disagreements[i]))
    plt.legend(set_percent)
    plt.ylabel('PD(L)')
    plt.savefig(fig_folder / "PDL_over_time", facecolor='w', transparent=False)
    plt.clf()

    print("Plotting Disagreement versus fixed percent")
    final_d = []
    for i in range(length):
      final_d.append(np.array(final_disagreements[i])[-1])
    plt.plot(set_percent, final_d, marker='o')
    plt.ylabel('Disagreement')
    plt.savefig(fig_folder / "disagreement_vs_nodes", facecolor='w', transparent=False)
    plt.clf()


if (__name__ == '__main__'):
    exp1(20,"exp1/")
    #exp1(40,"exp2/")
    #exp3("exp3/")
    #exp4("exp4/")
    #exp6("exp6/")

exit()

