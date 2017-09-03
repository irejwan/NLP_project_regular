import DFA.DFA as DFA
import pydot
from data_utils import get_data_alphabet
from main import init_state
from search_node import SearchNode
from state import State
from config import Config
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

config = Config()


def print_graph(nodes_list, graph_name):
    _, inv_alphabet_map = get_data_alphabet()
    graph = pydot.Dot(graph_type='digraph')
    nodes_dict = dict()
    i = 0

    for state in nodes_list:
        is_accept = state.is_accept
        color = 'green' if is_accept else 'red'
        nodes_dict[state] = pydot.Node(i, style="filled", fillcolor=color)
        graph.add_node(nodes_dict[state])
        i += 1

    for state in nodes_list:
        trans = state.transitions
        for input in trans.keys():
            next_state = trans[input]
            graph.add_edge(pydot.Edge(nodes_dict[state], nodes_dict[next_state], label=str(inv_alphabet_map[input])))
    graph.write_png(graph_name)


def minimize_dfa(nodes, start):
    """
    building a DFA automaton from the nodes, and returning an equal graph using MN algorithm.
    :param nodes: the original nodes.
    :return: the minimized graph's set of nodes.
    """
    fail_vec = init_state - 2  # a "garbage" state
    fail_state = SearchNode(State(fail_vec, quantized=tuple(fail_vec)))
    nodes.update({fail_state: {}})

    states = nodes
    # start = SearchNode(State(init_state, quantized=tuple(init_state)))
    accepts = [node for node in nodes if node.is_accept]
    alphabet = set()
    for node in nodes:  # making sure this is the alphabet that is actually being used
        alphabet.update(key for key in node.transitions)

    def delta(state, char):
        return nodes[state].get(char, fail_state)  # return fail state in case no transition is set

    d = DFA.DFA(states=states, start=start, accepts=accepts, alphabet=alphabet, delta=delta)
    eq_classes = d.minimize()
    for state_class in eq_classes:  # updating the nodes MN representatives
        representative = state_class[0]
        for node in state_class:
            node.representative = representative

    # update the node's transitions by the new delta function of the minimized DFA
    for node in d.states:
        new_transitions = {}
        for char in alphabet:
            new_transitions[char] = d.delta(node, char)
        node.transitions = new_transitions
    return d.states


def get_reverse_graph(nodes_dict):
    """
    returns the reverse graph of the nodes dict.
    for example, for the entry s1: {1, s2} - the output dict will contain: s2: {1, s1}
    :param nodes_dict: a dictionary that maps nodes to their transitions.
    :return: the reversed graph's dictionary.
    """
    flipped_nodes = {node: {} for node in nodes_dict}
    for state, trans in nodes_dict.items():
        for char, next_state in trans.items():
            if char in flipped_nodes[next_state]:  # in case two edges with the same char are directed to this node
                char += 10
            flipped_nodes[next_state][char] = state

    output_nodes = set()
    for node in flipped_nodes.keys():  # updating the nodes transitions
        node.transitions = flipped_nodes[node]
        output_nodes.add(node)
    return output_nodes


def bfs(start_node):
    visited, queue = set(), [start_node]
    while queue:
        vertex = queue.pop(0)
        if vertex not in visited:
            visited.add(vertex)
            vertex_neighbors = set(vertex.transitions[key] for key in vertex.transitions)
            queue.extend(vertex_neighbors - visited)
    return visited


def get_reachable_nodes(reversed_graph_nodes):
    """
    returns nodes that can be accessed by the accepting states.
    :param reversed_graph_nodes: the nodes of the reversed graph.
    :return: a set of reachable nodes.
    """
    accepting_nodes = [node for node in reversed_graph_nodes if node.is_accept]
    reachable = set()
    for node in accepting_nodes:
        reachable = set.union(reachable, bfs(node))
    return reachable


def copy(nodes_dict):
    """
    returns a dictionary that includes a copy of the nodes in the original nodes_dict.
    :param nodes_dict: a dictionary that maps every node to its transitions.
    :return: a copy of the original dict, such that it contains the same entries (with the new nodes)
    """
    output = {}
    old_to_new_states = {node: SearchNode(node.state) for node in nodes_dict}
    for state, trans in nodes_dict.items():
        if state not in output:
            output[old_to_new_states[state]] = {}
        for char, next_state in trans.items():
            output[old_to_new_states[state]][char] = old_to_new_states[next_state]
    return output


def get_trimmed_graph(analog_nodes):
    """
    trim the original graph - trim nodes that do not lead to any accepting node (not reachable).
    we do this by running bfs from the accepting nodes in the reversed graph.
    :param analog_nodes: all the nodes
    :return: all the nodes in the trimmed graph
    """
    graph_copy = copy(analog_nodes)
    assert len(graph_copy) == len(analog_nodes)
    reversed_graph_nodes = get_reverse_graph(graph_copy)
    reachable_nodes = get_reachable_nodes(reversed_graph_nodes)  # get all the states that lead to accepting states
    trimmed_graph = set()
    for node in analog_nodes:
        if node in reachable_nodes:
            trimmed_graph.add(node)
            new_trans = {char: next_node for char, next_node in node.transitions.items()
                         if next_node in reachable_nodes}
            node.transitions = new_trans  # updating the node's transitions so it will not lead to dead nodes
    assert len([node for node in trimmed_graph if node.is_accept]) == len(
        [node for node in analog_nodes if node.is_accept])
    return trimmed_graph


def retrieve_minimized_equivalent_graph(graph_nodes, graph_prefix_name, init_node, path='', plot=False):
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

    print_graph(trimmed_graph, path + graph_prefix_name + '_trimmed_graph.png')

    reduced_nodes = minimize_dfa({node: node.transitions for node in trimmed_graph}, init_node)
    print('num of nodes in the', graph_prefix_name, 'mn graph:', len(reduced_nodes))
    print_graph(reduced_nodes, path + graph_prefix_name + '_minimized_mn.png')

    if plot and len(trimmed_graph) > 0:
        all_nodes = list(trimmed_graph)  # we cast the set into a list, so we'll keep the order
        all_states = [node.state.vec for node in all_nodes]
        representatives = set([node.representative for node in trimmed_graph])
        representatives_colors_map = {rep: i for i, rep in enumerate(representatives)}
        colors = [representatives_colors_map[node.representative] for node in all_nodes]
        plot_states(all_states, colors)

    return trimmed_states


def evaluate_graph(X, y, init_node):
    acc = 0
    for sent, label in zip(X, y):
        curr_node = init_node
        for word in sent:
            curr_node = curr_node.transitions[word]
        prediction = curr_node.is_accept
        acc += (prediction == label)
    return acc / len(y)


def is_accurate(X, y, init_node):
    for sent, label in zip(X, y):
        curr_node = init_node
        for word in sent:
            curr_node = curr_node.transitions[word]
        prediction = curr_node.is_accept
        if prediction != label:
            return False
    return True


def plot_states(states, colors):
    le = PCA(n_components=2)
    le_X = le.fit_transform(states)
    plt.scatter(le_X[:, 0], le_X[:, 1], c=colors)
    plt.show()