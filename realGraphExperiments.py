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
    #Forming a distribution of edges and removig n of them
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
    #Forming a distribution of edges and removig n of them
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

def remap(G):
    mapping = {}
    id = 0
    for node in G:
        mapping[node] = id
        id += 1
    return nx.relabel_nodes(G, mapping, copy=True)

def expReal(rootFolder):
    #GET THE REAL GRAPHS LOADED INTO MEMORY
    data_folder = Path("realGraphData/")
    reddit_folder = Path("realGraphData/reddit/")
    twit_folder = Path("realGraphData/twitter-delhi2013/")

    fb_graph_list = data_folder / "facebook_combined.txt"
    reddit_graph_list = reddit_folder / "reddit_edgelist.csv"
    twit_graph_list = twit_folder / "edges_twitter.txt"

    reddit = open(reddit_graph_list, "r")
    twit = open(twit_graph_list, "r")

    fb = nx.read_edgelist(fb_graph_list, create_using=nx.Graph(),nodetype=int)
    fb.name = "facebook"
    fb = nx.k_core(fb, k=2)
    fb = remap(fb)
    print(nx.info(fb))

    re = nx.parse_edgelist(reddit, delimiter=',', create_using=nx.Graph(), nodetype=int)
    re.name = "reddit"
    re = nx.k_core(re, k=2)
    re = remap(re)
    print(nx.info(re))

    tw = nx.parse_edgelist(twit, delimiter='	', create_using=nx.Graph(), nodetype=int)
    tw.name = "twitter"
    tw = nx.k_core(tw, k=2)
    tw = remap(tw)
    print(nx.info(tw))

    iterations = 10000
    # innate_opinion = [np.random.normal(-0.7,0.2) for _ in range(int(n/2))]
    # innate_opinion2 = [np.random.normal(0.7,0.2) for _ in range(int(n/2))]
    # innate_opinion.extend(innate_opinion2)
    removal_edges = [0.1, 0.2, 0.3, 0.4, 0.5, 0.7]
    realgraphs = [tw]
    polarizations = collections.defaultdict(list)
    disagreements = collections.defaultdict(list)
    snapshots = collections.defaultdict(list)
    opinions = collections.defaultdict(list)

    for q,graph in tqdm(enumerate(realgraphs)):
        for o,rem in tqdm(enumerate(removal_edges)):
            G = graph.copy()

            n = G.number_of_nodes()
            identity = np.identity(n, dtype = float)
            innate_opinion = [random.uniform(-1,1) for _ in range(n)]

            A = nx.linalg.graphmatrix.adjacency_matrix(G)
            laplace = nx.linalg.laplacian_matrix(G)

            polarization = []
            disagreement = []
            modality = []
            G_over_time = []
            opinions_over_time = []
            # Apply FJ opinion model to get the first equilibrium state
            new_opinion = gf.change_fj_opinions(laplace, innate_opinion, identity)
            polarization.append(gf.calculate_polarization(new_opinion))
            disagreement.append(gf.calculate_disagreement(laplace, new_opinion))
            G_over_time.append(G.copy())
            opinions_over_time.append(innate_opinion)

            for i in tqdm(range(iterations)):
                r = remove_edges(G, new_opinion, int(n*rem))
                a = gf.add_new_friend_for_each_node(G,r,n)
                laplace = nx.linalg.laplacian_matrix(G)
                new_opinion = gf.change_fj_opinions(laplace, innate_opinion, identity)

                if (i % 1000 == 0):
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
    if not os.path.isdir(rootFolder + "/histograms"):
        os.makedirs(rootFolder + "/histograms")

    fig_folder = Path(rootFolder + "/histograms/" + str(graph.name) )
    if not os.path.isdir(fig_folder):
        os.makedirs(fig_folder)
    for iter,rem in enumerate(removal_edges):
        rem_folder = Path(fig_folder / str(rem))
        if not os.path.isdir(rem_folder):
            os.makedirs(rem_folder)
        for i in range(10):
            plt.hist(np.asnumpy(opinions[iter][0][i]), bins = numpy.arange(-1, 1, 0.05))
            plt.title("Histogram for removing edge percentage " + str(rem) + " @ " + str(i*1000) + " iterations")
            plt.savefig(rem_folder / str(i*1000), facecolor='w', transparent=False)
            plt.clf()

    final_polarizations = []
    final_disagreements = []
    final_modalities = []
    length = len(removal_edges)
    for i in range(length):
        final_polarizations.append(np.mean(np.array(polarizations[i]), axis = 0))
        final_disagreements.append(np.mean(np.array(disagreements[i]), axis = 0))
        #final_modalities.append(np.mean(np.array(modalities[i]), axis = 0))

    if not os.path.isdir(rootFolder + "/" + str(graph.name)):
        os.makedirs(rootFolder + "/" + str(graph.name))
    iterations_array = numpy.arange(iterations + 1)
    fig_folder = Path(rootFolder + "/" + str(graph.name))

    print("Polarization plot over time")
    for i in range(length):
        plt.plot(iterations_array, np.asnumpy(final_polarizations[i]))
    plt.legend(removal_edges)
    plt.ylabel('Polarization')
    plt.savefig(fig_folder / "polarization_over_time", facecolor='w', transparent=False)
    plt.clf()

    print("Plotting polarization versus Edge Removal")
    final_p = []
    for i in range(length):
        final_p.append(np.asnumpy(final_polarizations[i])[-1])
    plt.plot(removal_edges, final_p, marker='o')
    plt.ylabel('Polarization')
    plt.savefig(fig_folder / "polarization_vs_edges", facecolor='w', transparent=False)
    plt.clf()

    print("Disagreement plot over time")
    for i in range(length):
        plt.plot(iterations_array, np.asnumpy(final_disagreements[i]))
    plt.legend(removal_edges)
    plt.ylabel('Disagreement')
    plt.savefig(fig_folder / "disagreement_over_time", facecolor='w', transparent=False)
    plt.clf()

    print("Polarization&Disagreement plot over time")
    for i in range(length):
        plt.plot(iterations_array, np.asnumpy(final_polarizations[i])+np.asnumpy(final_disagreements[i]))
    plt.legend(removal_edges)
    plt.ylabel('PD(L)')
    plt.savefig(fig_folder / "PDL_over_time", facecolor='w', transparent=False)
    plt.clf()

    print("Plotting disagreement versus Edge Removal")
    final_d = []
    for i in range(length):
        final_d.append(np.asnumpy(final_disagreements[i])[-1])
    plt.plot(removal_edges, final_d, marker='o')
    plt.ylabel('Disagreement')
    plt.savefig(fig_folder / "disagreement_vs_edges", facecolor='w', transparent=False)
    plt.clf()

if (__name__ == '__main__'):
    expReal("realGraphs")

exit()
