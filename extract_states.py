from config import Config
from kmeans_handler import *
from meanshift_handler import *
from utils import *
import matplotlib.pyplot as plt

config = Config()


def get_graph(net, start, alphabet, kmeans_model=None):
    visited, queue = set(), [start]
    old_to_new_nodes = {start: start}
    while queue:
        vertex = queue.pop(0)
        if vertex not in visited:
            visited.add(vertex)
            vertex_neighbors = vertex.get_next_nodes(net, alphabet, old_to_new_nodes, kmeans_model)
            queue.extend(vertex_neighbors - visited)
    return visited


def quantize_states(analog_states, kmeans=None):
    for state in analog_states:
        quantize_state(state, kmeans)


def quantize_state(state, kmeans=None):
    if kmeans:
        state.quantized = get_cluster(state.vec, kmeans)
    else:
        state.quantized = State.quantize_vec(state.vec, config.States.intervals_num.int)


def get_analog_nodes(train_data, init_state, net):
    """
    get all possible states that the net returns for the training data.
    :param train_data: the training data
    :param init_state: the initial state (we start from this state for each input sentence)
    :return: all possible nodes, including transitions to the next nodes.
    """

    init_node = SearchNode(State(init_state, quantized=tuple(init_state)))
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


def quantize_graph(states_vectors_pool, init_state, net, train_data, alphabet_idx, max_k):
    """
    returns the nodes of the extracted graph, minimized by quantization.
    we merge the states by quantizing their vectors by using kmeans algorithm - the nodes are the centers
    returned by kmeans.
    after getting the centers we run BFS from the starting node to get their transitions.
    :param states_vectors_pool: all possible state vectors returned by the net.
    :param init_state: the initial state vector.
    :param max_k: maximum clusters allowed.
    :return: the nodes of the minimized graph.
    """
    states_vectors_pool = np.array(states_vectors_pool)

    clustering_model = config.ClusteringModel.model.str
    if clustering_model == 'k_means':
        trained_clustering_model = get_best_kmeans_model(states_vectors_pool, max_k)
    else:
        trained_clustering_model = get_best_meanshift_model(states_vectors_pool)
    print(trained_clustering_model.cluster_centers_.shape)
    # plt.scatter(states_vectors_pool[:, 0], states_vectors_pool[:, 1])
    # plt.scatter(trained_clustering_model.cluster_centers_[:, 0], trained_clustering_model.cluster_centers_[:, 1], c='r')
    # plt.draw()
    # plt.show()

    initial_state = State(init_state)
    analog_states = [initial_state] + \
                    [State(vec) for vec in states_vectors_pool if not np.array_equal(vec, init_state)]
    quantize_states(analog_states, kmeans=trained_clustering_model)
    start = SearchNode(initial_state)
    nodes = get_graph(net, start, alphabet_idx, kmeans_model=trained_clustering_model)

    start.state.final = net.is_accept(np.array([start.state.vec]))
    for sent in train_data:
        curr_node = start
        for word in sent:
            next_node = curr_node.transitions[word]
            curr_node = next_node
        curr_node.state.final = net.is_accept(np.array([curr_node.state.vec]))
    return {node: node.transitions for node in nodes}, start


def retrieve_minimized_equivalent_graph(graph_nodes, graph_prefix_name, init_node):
    trimmed_graph = get_trimmed_graph(graph_nodes)
    print('num of nodes in the', graph_prefix_name, 'trimmed graph:', len(trimmed_graph))
    print_graph(trimmed_graph, graph_prefix_name + '_trimmed_graph.png')

    reduced_nodes = minimize_dfa({node: node.transitions for node in trimmed_graph}, init_node)
    print('num of nodes in the', graph_prefix_name, 'mn graph:', len(reduced_nodes))
    print_graph(reduced_nodes, graph_prefix_name + '_minimized_mn.png')
