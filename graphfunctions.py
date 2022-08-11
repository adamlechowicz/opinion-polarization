#Opinion Dynamics, Polarization, Disagreement
import sys
import numpy as np
import numpy
import scipy as sp
import networkx as nx
import math
import random
from random import randrange

def change_fj_opinions(laplace, innate_opinion, identity):

    cupy_l = np.array(laplace.toarray())
    li = np.array(cupy_l + identity)
    new_opinion = np.linalg.solve(li, np.array(innate_opinion))

    return new_opinion.reshape(-1)


def calculate_polarization(opinions):

    return opinions.var() * len(opinions)


def calculate_disagreement(laplace, opinions):

    #l = np.array(laplace.toarray())
    return np.dot((opinions.T * laplace), opinions)


#FOF Recommender
import random


def friends_of_friends_set(G, friends):
    friends_of_friends = set()

    for f in friends:
        friends_of_friends.update(G.neighbors(f))

    return friends_of_friends


def make_remove_set(G, node, friends_of_friends):
    """Removes friends of friends that a user is already friends with"""
    remove_set = set()

    for fof in friends_of_friends:
        if G.has_edge(node, fof):
            remove_set.add(fof)

    return remove_set


def find_friends_of_friends(G, node):
    # Find the list of friends of friends who the user is not friends with

    friends_of_friends = friends_of_friends_set(G, G.neighbors(node))
    to_remove = make_remove_set(G, node, friends_of_friends)
    friends_of_friends.difference_update(to_remove)
    return friends_of_friends


def suggest_Friend(G, node):
    fof = find_friends_of_friends(G, node)

    if node in fof:
        fof.remove(node)

    if not fof:
        return None

    return list(fof)[random.randint(0, len(fof) - 1)]


def add_new_friend_for_each_node(G, to_add, n):
    new_edges_to_add = set()
    
    while len(new_edges_to_add) < to_add:
        node = random.randint(0,n - 1)
        ftA = suggest_Friend(G, node)
        if ftA:
            if not ((ftA, node) in new_edges_to_add or (node, ftA) in new_edges_to_add):
                new_edges_to_add.add((node, ftA))

    G.add_edges_from(new_edges_to_add)
    

def add_new_RANDOM_friend_for_each_node(G, to_add, n):
    new_edges_to_add = set()
    edges_list = set(G.edges)

    while len(new_edges_to_add) < to_add:
        node = random.randint(0, n - 1)
        ftA = random.randint(0, n - 1)
        if ftA != node:
            if not ((ftA, node) in new_edges_to_add or (node, ftA) in new_edges_to_add) or (ftA, node) in edges_list or (node, ftA) in edges_list:
                new_edges_to_add.add((node, ftA))

    G.add_edges_from(new_edges_to_add)

