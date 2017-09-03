import time
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

    def quantize_state(state, model=None):
        if model:
            state.quantized = get_cluster(state.vec, model)
        else:
            state.quantized = State.quantize_vec(state.vec, config.States.intervals_num.int)

    for state in analog_states:
        quantize_state(state, model)


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


def get_quantized_graph(analog_states, init_node, net, X, y, plot=False):
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

    states_vectors_pool = np.array([state.vec for state in analog_states if state != init_node.state])
    clustering_model = config.ClusteringModel.model.str
    if clustering_model == 'k_means':
        cluster_model = get_kmeans(analog_states, init_node, net, X, y)
    else:
        cluster_model = get_best_meanshift_model(states_vectors_pool)

    if plot:
        plot_states(states_vectors_pool, cluster_model.predict(states_vectors_pool))

    nodes = get_quantized_graph_for_model(alphabet_idx, analog_states, cluster_model, init_node, net, X)
    return {node: node.transitions for node in nodes}


def get_quantized_graph_for_model(alphabet_idx, analog_states, cluster_model, init_node, net, train_data):
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
    nodes = get_graph(net, init_node, alphabet_idx, model=cluster_model)
    # init_node.state.final = net.is_accept(np.array([init_node.state.vec]))
    for sent in train_data:
        curr_node = init_node
        for word in sent:
            next_node = curr_node.transitions[word]
            curr_node = next_node
        curr_node.state.final = net.is_accept(np.array([curr_node.state.vec]))
    return nodes


def get_kmeans(analog_states, init_node, net, X, y, min_k=50, acc_th=0.9):
    _, alphabet_idx = get_data_alphabet()
    print('working on k-means')
    size = len(analog_states) - 1
    factor = int(np.log(size))
    curr_k = min_k

    states_vectors_pool = np.array([state.vec for state in analog_states])
    adeq = 0

    while adeq < acc_th and curr_k < size:
        adeq, curr_model = evaluate_kmeans_model(curr_k, alphabet_idx, analog_states, init_node, net, X, y)
        curr_k *= factor

    if adeq < acc_th:
        max_k = size
        min_k = curr_k // factor
    else:
        max_k = curr_k // factor
        min_k = max_k // factor

    print('k_max = {} and k_min = {}'.format(max_k, min_k))

    while max_k - min_k > factor:
        curr_k = min_k + (max_k - min_k) // 2
        adeq, curr_model = evaluate_kmeans_model(curr_k, alphabet_idx, analog_states, init_node, net, X, y)
        if adeq >= acc_th:
            max_k = curr_k
        else:
            min_k = curr_k

    k = min_k + (max_k - min_k) // 2
    print('finished. best k is:', k)
    return KMeans(n_clusters=k).fit(states_vectors_pool)


def evaluate_kmeans_model(k, alphabet_idx, analog_states, init_node, net, X, y):
    print('k = {}:'.format(k), end=' ')
    clk = time.clock()
    states_vectors_pool = np.array([state.vec for state in analog_states])
    curr_model = KMeans(n_clusters=k, n_jobs=5, algorithm='elkan', max_iter=20).fit(states_vectors_pool)
    get_quantized_graph_for_model(alphabet_idx, analog_states, curr_model, init_node, net, X)
    adeq = evaluate_graph(X, y, init_node)
    clk2 = time.clock()
    print('took {:.2f} sec,'.format(clk2 - clk), 'adequate to the net in', adeq, 'of validation sentences')
    return adeq, curr_model

