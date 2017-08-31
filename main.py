from extract_states import *
from utils import *
from data_utils import generate_sentences
from regular_rnn import RegularRNN
import tensorflow as tf
from config import Config
from sklearn.cluster import KMeans

config = Config()

num_sents = config.Data.num_sents.int
NUM_EPOCHS = config.RNN.NUM_EPOCHS.int
state_size = config.RNN.state_size.int
init_state = np.zeros(state_size)


def train(X_train, y_train, X_test, y_test, sess, rnn):
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter('board', sess.graph)
    correct = []
    correct_labels = []

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

            if epoch == NUM_EPOCHS - 1 and acc == 1:
                correct.append(sentence)
                correct_labels.append(label)

        accuracy_test /= len(y_test)
        loss_test /= len(y_test)

        print("loss: {:.3f}, val_loss: {:.3f}, acc: {:.3f}, val_acc: {:.3f}".format(
            loss_train, loss_test, accuracy_train, accuracy_test
        ))

    train_writer.close()
    return correct, correct_labels


def get_kmeans(states_vector_pool, init_state, net, X, y, alphabet_idx, min_k=1):
    print('working on k-means')
    curr_k = min_k
    curr_model = KMeans(n_clusters=curr_k).fit(states_vector_pool)
    quantized_nodes, init_node = get_quantized_graph(states_vector_pool, init_state, net, X, alphabet_idx, curr_model)
    acc = evaluate_graph(X, y, init_node)
    history = []
    while acc < 1:
        curr_k *= 2
        curr_model = KMeans(n_clusters=curr_k).fit(states_vector_pool)
        quantized_nodes, init_node = get_quantized_graph(states_vector_pool, init_state, net, X, alphabet_idx, curr_model)
        acc = evaluate_graph(X, y, init_node)
        history.append((curr_k, acc))
        print('k =', curr_k, 'acc =', acc)
    max_k = curr_k

    print('k_max = {} and k_min = {}'.format(max_k, min_k))
    history = []
    while max_k - min_k > 1:
        curr_k = min_k + (max_k - min_k) // 2
        curr_model = KMeans(n_clusters=curr_k).fit(states_vector_pool)
        quantized_nodes, init_node = get_quantized_graph(states_vector_pool, init_state, net, X, alphabet_idx, curr_model)
        acc = evaluate_graph(X, y, init_node)
        history.append((curr_k, acc))
        print('k =', curr_k, 'acc =', acc)
        if acc == 1:
            max_k = curr_k
        else:
            min_k = curr_k
    print('finished. best k is:', curr_k)
    plt.scatter(*zip(*history))
    plt.show()
    return curr_model


def extract_graphs(X, y, inv_alphabet_map):
    """
    after the net has trained enough, we calculate the states returned and print the graphs of them.
    :param X: the training data
    :param y: the labels
    :return: nothing
    """
    init_node = SearchNode(State(init_state, quantized=tuple(init_state)))
    analog_nodes = get_analog_nodes(X, init_node, rnn)

    for node in analog_nodes:
        node.transitions = analog_nodes[node]
    # print_graph(analog_nodes, 'orig.png')
    print('num of nodes in original graph:', len(analog_nodes))
    mn_size = retrieve_minimized_equivalent_graph(analog_nodes, 'orig', init_node, inv_alphabet_map)

    states_vectors_pool = [node.state.vec for node in analog_nodes]
    cluster_model = get_kmeans(states_vectors_pool, init_state, rnn, X, y,
                               list(inv_alphabet_map.keys()), min_k=mn_size)
    print(cluster_model.cluster_centers_.shape)
    le = PCA(n_components=2)
    le_X = le.fit_transform(states_vectors_pool)
    plt.scatter(le_X[:, 0], le_X[:, 1], c=cluster_model.predict(states_vectors_pool))
    plt.show()
    quantized_nodes, init_node = get_quantized_graph(states_vectors_pool, init_state, rnn,
                                                     X, list(inv_alphabet_map.keys()), cluster_model)
    acc = evaluate_graph(X, y, init_node)
    print('quantized graph is correct in {:.1f}% of the sentences classified correctly by the RNN'.format(acc*100))
    print_graph(quantized_nodes, 'quantized_graph_reduced.png', inv_alphabet_map)
    retrieve_minimized_equivalent_graph(quantized_nodes, 'quantized', init_node, inv_alphabet_map)


if __name__ == '__main__':
    alphabet = config.Grammar.alphabet.lst
    alphabet_map = {a: i for i, a in enumerate(alphabet)}
    inv_alphabet_map = {v: k for k, v in alphabet_map.items()}

    # print(alphabet_map)

    X_train, y_train, X_test, y_test = generate_sentences(num_sents, alphabet_map)
    sess = tf.InteractiveSession()
    rnn = RegularRNN(sess)
    sess.run(tf.global_variables_initializer())
    correct, correct_labels = train(X_train, y_train, X_test, y_test, sess, rnn)
    print('num of strings classified correctly by the net: ', len(correct))
    extract_graphs(correct, correct_labels, inv_alphabet_map)
