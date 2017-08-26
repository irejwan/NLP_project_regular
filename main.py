from extract_states import *
from utils import *
from data_utils import generate_sentences
from regular_rnn import RegularRNN
import tensorflow as tf
import matplotlib.pyplot as plt
from config import Config

config = Config()


DATA_AMOUNT = config.RNN.DATA_AMOUNT.int
NUM_EPOCHS = config.RNN.NUM_EPOCHS.int

state_size = config.RNN.state_size.int
init_state = np.zeros(state_size)


def train(X_train, y_train, X_test, y_test, sess, rnn):
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter('board', sess.graph)
    errors = list()

    print("Start learning...")
    for epoch in range(NUM_EPOCHS):
        loss_train = 0
        accuracy_train = 0
        i = 0

        print("epoch: {}\t".format(epoch), end="")

        # Training
        for sentence, label in zip(X_train, y_train):
            acc, loss_tr, _, summary = sess.run([rnn.accuracy, rnn.loss, rnn.optimizer, merged],
                                                feed_dict={rnn.label_ph: label,
                                                           rnn.init_state_ph: [init_state],
                                                           rnn.input_ph: sentence})

            accuracy_train += acc
            loss_train += loss_tr
            train_writer.add_summary(summary, i)
            i += 1
        accuracy_train /= len(y_train)
        loss_train /= len(y_train)

        # Testing
        accuracy_test = 0
        loss_test = 0

        for sentence, label in zip(X_test, y_test):
            loss, acc = sess.run([rnn.loss, rnn.accuracy],
                                 feed_dict={rnn.label_ph: label, rnn.input_ph: sentence,
                                            rnn.init_state_ph: [init_state]})
            accuracy_test += acc
            loss_test += loss

            if (epoch == NUM_EPOCHS - 1) and (acc == 0):  # last epoch
                errors.append(sentence)

        accuracy_test /= len(y_test)
        loss_test /= len(y_test)

        print("loss: {:.3f}, val_loss: {:.3f}, acc: {:.3f}, val_acc: {:.3f}".format(
            loss_train, loss_test, accuracy_train, accuracy_test
        ))

    train_writer.close()
    return errors


def morco(X):
    analog_nodes, states_pool = get_analog_nodes(X, rnn, init_state)
    analog_nodes_states = analog_nodes.keys()
    for node in analog_nodes:
        node.transitions = analog_nodes[node]
    print_graph(analog_nodes_states, 'graph.png')
    print('num of nodes in original graph:', len(analog_nodes_states))

    graph_copy = copy(analog_nodes)
    assert len(graph_copy) == len(analog_nodes)
    reversed_graph_nodes = get_reverse_graph(graph_copy)
    print_graph(reversed_graph_nodes, 'reversed_graph.png')
    reachable_nodes = get_reachable_nodes(reversed_graph_nodes)
    trimmed_graph = set()
    for node in set(analog_nodes_states):
        if node in reachable_nodes:
            trimmed_graph.add(node)
            new_trans = {char: next_node for char, next_node in node.transitions.items()
                         if next_node in reachable_nodes}
            node.transitions = new_trans  # updating the node's transitions so it will not lead to dead nodes
    assert len([node for node in trimmed_graph if node.is_accept]) == len([node for node in analog_nodes if node.is_accept])
    print_graph(trimmed_graph, 'trimmed_graph.png')
    print('num of nodes in trimmed graph:', len(trimmed_graph))

    reduced_nodes = minimize_dfa({node: node.transitions for node in trimmed_graph})
    print_graph(reduced_nodes, 'graph_minimized_mn.png')
    print('num of nodes in mn graph:', len(reduced_nodes))

    states_pool = np.array(states_pool)

    kmeans_model = get_best_kmeans_model(states_pool, len(reduced_nodes) + 1)
    analog_states = [State(vec) for vec in states_pool]
    quantize_states(analog_states, kmeans=kmeans_model)
    nodes_list = get_graph(rnn, analog_states[0], [1, 2], kmeans_model=kmeans_model)

    for node in nodes_list:
        node.state.final = rnn.is_accept([node.state.quantized])

    print_graph(nodes_list, 'graph_reduced.png')

    print(len(analog_states))
    print(kmeans_model.cluster_centers_.shape)
    plt.scatter(states_pool[:, 0], states_pool[:, 1])
    plt.scatter(kmeans_model.cluster_centers_[:, 0], kmeans_model.cluster_centers_[:, 1], c='r')
    plt.draw()
    plt.show()

    for state in nodes_list:
        is_accept = rnn.is_accept([state.state.quantized])
        print(is_accept)
        print(state)


def test():
    states = [SearchNode(State([i])) for i in range(7)]
    s0, s1, s2, s3, s4, s5, s6 = states
    s0.transitions = {1: s2, 2: s1}
    s1.transitions = {2: s6}
    s2.transitions = {2: s3}
    s3.transitions = {1: s4}
    s4.transitions = {2: s5}
    s5.state.final = True
    states_dict = {s0: s0.transitions, s1: s1.transitions, s2: s2.transitions, s3: s3.transitions,
                   s4: s4.transitions, s5: s5.transitions, s6: s6.transitions}
    reversed_graph_nodes = get_reverse_graph(copy(states_dict))
    reachable_nodes = get_reachable_nodes(reversed_graph_nodes)
    trimmed_graph = set.intersection(set(states_dict.keys()), reachable_nodes)
    print(len(trimmed_graph))


if __name__ == '__main__':
    # test()
    X_train, y_train, X_test, y_test = generate_sentences(DATA_AMOUNT)
    sess = tf.InteractiveSession()
    rnn = RegularRNN(sess)
    sess.run(tf.global_variables_initializer())
    train(X_train, y_train, X_test, y_test, sess, rnn)
    morco(X_train)
