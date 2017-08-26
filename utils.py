import DFA.DFA as DFA
import pydot
from main import init_state
from search_node import SearchNode
from state import State


def print_graph(nodes_list, graph_name):
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
            if next_state not in nodes_dict:
                is_accept = next_state.is_accept
                color = 'green' if is_accept else 'red'
                nodes_dict[next_state] = pydot.Node(i, style="filled", fillcolor=color)
                graph.add_node(nodes_dict[next_state])
                i += 1
            graph.add_edge(pydot.Edge(nodes_dict[state], nodes_dict[next_state], label=str(int(input))))
    graph.write_png(graph_name)


def minimize_dfa(node_objects):
    fail_vec = init_state - 2  # added a "garbage" state
    fail_state = SearchNode(State(fail_vec, quantized=tuple(fail_vec)))
    node_objects.update({fail_state: {}})

    states = node_objects
    start = SearchNode(State(init_state, quantized=tuple(init_state)))
    accepts = [node for node in node_objects if node.is_accept]
    alphabet = [1, 2]

    def delta(state, char):
        return node_objects[state].get(char, fail_state)  # return fail state in case no transition is set

    d = DFA.DFA(states=states, start=start, accepts=accepts, alphabet=alphabet, delta=delta)
    d.minimize()

    # update the node's transitions by the new delta function of the minimized DFA
    for node in d.states:
        new_transitions = {}
        for char in alphabet:
            new_transitions[char] = d.delta(node, char)
        node.transitions = new_transitions
    return d.states


def get_reverse_graph(analog_nodes):
    flipped_nodes = {node: {} for node in analog_nodes}
    for state, trans in analog_nodes.items():
        for char, next_state in trans.items():
            if char in flipped_nodes[next_state]:  # in two edges with the same char are directed to this node
                char += 10
            flipped_nodes[next_state][char] = state

    output_nodes = set()
    for node in flipped_nodes.keys():
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
    accepting_nodes = [node for node in reversed_graph_nodes if node.is_accept]
    reachable = set()
    for node in accepting_nodes:
        reachable = set.union(reachable, bfs(node))
    return reachable


def copy(states_dict):
    output = {}
    old_to_new_states = {state: SearchNode(state.state) for state in states_dict}
    for state, trans in states_dict.items():
        if state not in output:
            output[old_to_new_states[state]] = {}
        for char, next_state in trans.items():
            output[old_to_new_states[state]][char] = old_to_new_states[next_state]
    return output

