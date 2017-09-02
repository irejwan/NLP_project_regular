import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

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


def get_clustering_model(states_vectors_pool, init_state, net, X, y):
    states_vectors_pool = np.array(states_vectors_pool)
    clustering_model = config.ClusteringModel.model.str
    if clustering_model == 'k_means':
        cluster_model = get_kmeans(states_vectors_pool, init_state, net, X, y)
    else:
        cluster_model = get_best_meanshift_model(states_vectors_pool)

    print(cluster_model.cluster_centers_.shape)
    plot_states(states_vectors_pool, cluster_model.predict(states_vectors_pool), plot=False)
    return cluster_model


def get_quantized_graph(states_vectors_pool, init_state, net, train_data, labels):
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
    analog_states = [State(vec) for vec in states_vectors_pool if
                     not np.array_equal(vec, init_state)]
    cluster_model = get_clustering_model(states_vectors_pool, init_state, net, train_data, labels) \
        if config.States.use_model.boolean else None
    nodes, start = get_quantized_graph_for_model(alphabet_idx, analog_states, cluster_model, init_state, net,
                                                 train_data)
    return {node: node.transitions for node in nodes}, start


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
    nodes = get_graph(net, start, alphabet_idx, model=cluster_model)
    start.state.final = net.is_accept(np.array([start.state.vec]))
    for sent in train_data:
        curr_node = start
        for word in sent:
            next_node = curr_node.transitions[word]
            curr_node = next_node
        curr_node.state.final = net.is_accept(np.array([curr_node.state.vec]))
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

    if len(trimmed_graph) > 300:
        print('trimmed graph too big, skipping MN')
        return 1

    print_graph(trimmed_graph, graph_prefix_name + '_trimmed_graph.png')

    reduced_nodes = minimize_dfa({node: node.transitions for node in trimmed_graph}, init_node)
    print('num of nodes in the', graph_prefix_name, 'mn graph:', len(reduced_nodes))
    print_graph(reduced_nodes, graph_prefix_name + '_minimized_mn.png')

    if plot and len(trimmed_graph) > 0:
        all_nodes = list(trimmed_graph)  # we cast the set into a list, so we'll keep the order
        all_states = [node.state.vec for node in all_nodes]
        representatives = set([node.representative for node in trimmed_graph])
        representatives_colors_map = {rep: i for i, rep in enumerate(representatives)}
        colors = [representatives_colors_map[node.representative] for node in all_nodes]
        plot_states(all_states, colors, plot)


def get_kmeans(states_vectors_pool, init_state, net, X, y, min_k=1):
    _, alphabet_idx = get_data_alphabet()
    analog_states = [State(vec) for vec in states_vectors_pool if
                     not np.array_equal(vec, init_state)]
    print('working on k-means')
    size = len(states_vectors_pool)
    factor = int(np.log(size))
    curr_k = min_k / factor
    acc = 0
    while acc < 1 and curr_k < size:
        curr_k = min(int(curr_k * factor), size)
        acc, curr_model = evaluate_current_model(X, alphabet_idx, analog_states, curr_k, init_state, net,
                                                 states_vectors_pool, y)
        print('k =', curr_k, 'acc =', acc)
    if curr_k == size:
        return KMeans(n_clusters=curr_k).fit(states_vectors_pool)

    min_k = curr_k // factor
    max_k = curr_k
    print('k_max = {} and k_min = {}'.format(max_k, min_k))

    while max_k - min_k > 1:
        curr_k = min_k + (max_k - min_k) // 2
        acc, curr_model = evaluate_current_model(X, alphabet_idx, analog_states, curr_k, init_state, net,
                                                 states_vectors_pool, y)
        print('k =', curr_k, 'acc =', acc)
        if acc == 1:
            max_k = curr_k
        else:
            min_k = curr_k
    print('finished. best k is:', curr_k)
    return curr_model


def evaluate_current_model(X, alphabet_idx, analog_states, curr_k, init_state, net, states_vectors_pool, y):
    curr_model = KMeans(n_clusters=curr_k).fit(states_vectors_pool)
    quantized_nodes, init_node = get_quantized_graph_for_model(alphabet_idx, analog_states, curr_model, init_state, net,
                                                               X)
    acc = evaluate_graph(X, y, init_node)
    return acc, curr_model


def plot_states(states, colors, plot=False):
    if plot:
        le = PCA(n_components=2)
        le_X = le.fit_transform(states)
        plt.scatter(le_X[:, 0], le_X[:, 1], c=colors)
        plt.show()
