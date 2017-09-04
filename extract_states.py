import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn import cluster

from clustering_handler import get_cluster, KMeans, get_best_meanshift_model
from utils import *
import numpy as np

config = Config()


def get_graph(net, start, alphabet, model=None):
    """
    running BFS - to get *all transitions for every node* which is reachable from the start.
    :param net: the RNN
    :param start: the starting node
    :param alphabet: the alphabet on the transitions
    :param model: quantization model
    :return: the graph's nodes reachable from the start
    """
    visited, queue = set(), [start]
    old_to_new_nodes = {start: start}
    while queue:
        vertex = queue.pop(0)
        if vertex not in visited:
            visited.add(vertex)
            vertex_neighbors = vertex.get_next_nodes(net, alphabet, old_to_new_nodes, model)
            queue.extend(vertex_neighbors - visited)
    return visited


def quantize_states(analog_states, model=None):
    """
    quantize an iterable of states by some quantization method.
    :param analog_states: a list of states
    :param model: if None - intervals quantization is used, else - a clustering model
    :return: Nothing
    """
    for state in analog_states:
        quantize_state(state, model)


def quantize_state(state, model=None):
    if model:
        state.quantized = get_cluster(state.vec, model)
    else:
        state.quantized = State.quantize_vec(state.vec, config.States.intervals_num.int)


def get_analog_nodes(train_data, init_node, net):
    """
    get all possible states that the net returns for the training data.
    :param train_data: the training data
    :param init_node: the initial state (we start from this state for each input sentence)
    :return: all possible nodes, including transitions to the next nodes.
    """

    init_node.state.final = net.is_accept(np.array([init_node.state.vec]))
    analog_nodes = {init_node: {}}
    for sent in train_data:
        curr_node = init_node
        for word in sent:
            next_state_vec = np.array(net.get_next_state(curr_node.state.vec, word))
            next_node = SearchNode(State(next_state_vec, quantized=tuple(next_state_vec)))
            if next_node not in analog_nodes:  # we do this in order to make sure *all* states are in analog_nodes
                analog_nodes[next_node] = {}
            analog_nodes[curr_node][word] = next_node
            curr_node = next_node
        curr_node.state.final = net.is_accept(np.array([curr_node.state.vec]))
    return analog_nodes


def get_clustering_model(analog_states, init_state, net, X, y):
    states_vectors_pool = np.array([state.vec for state in analog_states])
    clustering_model = config.ClusteringModel.model.str
    if clustering_model == 'k_means':
        # cluster_model = get_kmeans(analog_states, init_state, net, X, y)
        k = int(len(analog_states) ** 0.6)
        states = [state.vec for state in analog_states]
        cluster_model = KMeans(n_clusters=k).fit(states)
        # cluster_model = cluster.AffinityPropagation().fit(states)
    else:
        cluster_model = get_best_meanshift_model(states_vectors_pool)

    print(cluster_model.cluster_centers_.shape)
    plot_states(states_vectors_pool, cluster_model.predict(states_vectors_pool))
    return cluster_model


def get_quantized_graph(analog_states, init_state, net, train_data, labels, analog_nodes):
    """
    returns the nodes of the extracted graph, minimized by quantization.
    we merge the states by quantizing their vectors by using kmeans/meanshift algorithm - the nodes are the centers
    returned by kmeans.
    after getting the centers we run BFS from the starting node to get their transitions.
    :param states_vectors_pool: all possible state vectors returned by the net.
    :param init_state: the initial state vector.
    :param net: the RNN
    :param train_data: X
    :param labels: y
    :return: the nodes of the minimized graph.
    """
    _, alphabet_idx = get_data_alphabet()
    cluster_model = get_clustering_model(analog_states, init_state, net, train_data, labels) \
        if config.States.use_model.boolean else None
    nodes, start = get_quantized_graph_for_model(alphabet_idx, analog_states, cluster_model, init_state, net,
                                                 train_data)
    return {node: node.transitions for node in nodes}, start


def get_merged_graph_by_clusters(init_node, net, train_data, model):
    clustered_graph = {init_node: {}}
    nodes_map = {init_node: init_node}
    init_node.state.final = net.is_accept(np.array([init_node.state.vec]))
    for sent in train_data:
        curr_node = nodes_map[init_node]
        for word in sent:
            next_state_vec = np.array(net.get_next_state(curr_node.state.vec, word))
            next_node = SearchNode(State(next_state_vec, quantized=get_cluster(next_state_vec, model)))
            if next_node not in clustered_graph:
                clustered_graph[next_node] = {}
            if next_node not in nodes_map:
                nodes_map[next_node] = next_node
            curr_node = nodes_map[curr_node]
            next_node = nodes_map[next_node]
            if word in clustered_graph[curr_node] and clustered_graph[curr_node][word] != next_node:
                print("conflict")
                word += 10
            clustered_graph[curr_node][word] = next_node
            curr_node = next_node
        curr_node.state.final = net.is_accept(np.array([curr_node.state.vec]))
    for node in clustered_graph:
        node.transitions = clustered_graph[node]
    return clustered_graph


def get_quantized_graph_for_model(alphabet_idx, analog_states, cluster_model, init_state, net, train_data):
    """
    :param alphabet_idx:  the alphabet indices
    :param analog_states: a list of analog (continuous, non-quantized) States
    :param cluster_model: a specific clustering model - like k-means/meanshift or None (for interval quantization)
    :param init_state: the initial state vector
    :param net: the RNN
    :param train_data: the training data
    :return: the quantized graph by a specific model (cluster_model)
    """
    quantize_states(analog_states, model=cluster_model)
    start = SearchNode(State(init_state))
    # nodes = get_graph(net, start, alphabet_idx, model=cluster_model)
    nodes = get_merged_graph_by_clusters(start, net, train_data, model=cluster_model)
    return nodes, start


def retrieve_minimized_equivalent_graph(graph_nodes, graph_prefix_name, init_node, plot=False):
    """
    returns the exact equivalent graph using MN algorithm.
    complexity improvement: we trim the graph first - meaning, we only keep nodes that lead to an accepting state.
    :param graph_nodes: the original graph nodes
    :param graph_prefix_name: prefix name, for the .png file
    :param init_node: the initial state node
    :return: nothing
    """
    trimmed_graph = get_trimmed_graph(graph_nodes)
    print('num of nodes in the', graph_prefix_name, 'trimmed graph:', len(trimmed_graph))
    trimmed_states = [node.state for node in trimmed_graph]

    if len(trimmed_graph) > 300:
        print('trimmed graph too big, skipping MN')
        return trimmed_states

    print_graph(trimmed_graph, graph_prefix_name + '_trimmed_graph.png', init_node)

    reduced_nodes = minimize_dfa({node: node.transitions for node in trimmed_graph}, init_node)
    print('num of nodes in the', graph_prefix_name, 'mn graph:', len(reduced_nodes))
    print_graph(reduced_nodes, graph_prefix_name + '_minimized_mn.png', init_node)

    if plot and len(trimmed_graph) > 0:
        all_nodes = list(trimmed_graph)  # we cast the set into a list, so we'll keep the order
        all_states = [node.state.vec for node in all_nodes]
        representatives = set([node.representative for node in trimmed_graph])
        representatives_colors_map = {rep: i for i, rep in enumerate(representatives)}
        colors = [representatives_colors_map[node.representative] for node in all_nodes]
        plot_states(all_states, colors)

    return trimmed_states


def get_kmeans(analog_states, init_state, net, X, y, min_k=1):
    _, alphabet_idx = get_data_alphabet()
    print('working on k-means')
    size = len(analog_states) - 1
    factor = int(np.log(size))
    curr_k = min_k / factor

    accept_vecs = [state.vec for state in analog_states if state.final]
    reject_vecs = [state.vec for state in analog_states if not state.final]
    states_vectors_pool = np.array(accept_vecs + reject_vecs)
    acc = 0
    while acc < 1 and curr_k < size:
        curr_k = min(int(curr_k * factor), size)
        acc, curr_model = evaluate_current_model(curr_k, states_vectors_pool, accept_vecs, reject_vecs)
        print('k =', curr_k, 'acc =', acc)
    if curr_k == size:
        return curr_model

    min_k = curr_k // factor
    max_k = curr_k
    print('k_max = {} and k_min = {}'.format(max_k, min_k))

    while max_k - min_k > 1:
        curr_k = min_k + (max_k - min_k) // 2
        acc, curr_model = evaluate_current_model(curr_k, states_vectors_pool, accept_vecs, reject_vecs)
        print('k =', curr_k, 'acc =', acc)
        if acc == 1:
            max_k = curr_k
        else:
            min_k = curr_k
    print('finished. best k is:', curr_k)
    return curr_model


def evaluate_current_model(curr_k, states_vectors_pool, accept_vecs, reject_vecs):
    curr_model = KMeans(n_clusters=curr_k).fit(states_vectors_pool)
    accept_clusters = set(curr_model.predict(accept_vecs))
    reject_clusters = set(curr_model.predict(reject_vecs))
    if len(accept_clusters.intersection(reject_clusters)) > 0:
        return 0, None
    return 1, curr_model

    # quantized_nodes, init_node = get_quantized_graph_for_model(alphabet_idx, analog_states, curr_model, init_state, net,
    #                                                            X)
    # is_acc = is_accurate(X, y, init_node)
    # return is_acc, curr_model


def plot_states(states, colors):
    le = PCA(n_components=2)
    le_X = le.fit_transform(states)
    plt.scatter(le_X[:, 0], le_X[:, 1], c=colors)
    plt.show()


