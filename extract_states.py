from config import Config
from kmeans_handler import *
from search_node import SearchNode
from state import State

config = Config()


def get_graph(net, initial_state, alphabet, kmeans_model=None):
    start = SearchNode(initial_state)
    visited, queue = set(), [start]
    while queue:
        vertex = queue.pop(0)
        if vertex not in visited:
            visited.add(vertex)
            vertex_neighbors = vertex.get_next_nodes(net, alphabet, kmeans_model)
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


def get_analog_nodes(X, rnn, init_state):
    states_pool = [init_state]
    analog_nodes = {}
    for sent in X:
        curr_state = init_state
        curr_node = SearchNode(State(curr_state, quantized=tuple(curr_state)))
        if curr_node not in analog_nodes:
            analog_nodes[curr_node] = {}
        for word in sent:
            curr_node.state.final = rnn.is_accept(np.array([curr_node.state.vec]))
            next_state_vec = np.array(rnn.get_next_state(curr_node.state.vec, word))
            next_node = SearchNode(State(next_state_vec, quantized=tuple(next_state_vec)))
            if next_node not in analog_nodes:  # we do this in order to make sure *all* states are in analog_nodes
                analog_nodes[next_node] = {}
            states_pool.append(next_state_vec)
            analog_nodes[curr_node][word] = next_node
            curr_node = next_node

    return analog_nodes, states_pool

