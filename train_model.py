from lib.imports_n_methods_2019 import *

client = MongoClient()
db = client.employmentdb
collection = db.nzera
nlp = spacy.load('en_core_web_sm')
# Corpus and Train-Test split
corpus = "/home/moose/Documents/Education/allText.txt" 
incl_det = False # Whether to include determinations within the data
TRAIN_DATA = 175 
TOTAL_DATA = 218 
# Neural Network
num_epochs = 150 
batch_size = 50 
embedding_size = 128 
no_cells = 1 # Define the number of cells in the MultiRNNCell
max_label = 2 
num_units = [embedding_size] * no_cells # A list of no_cells cells of embedding_size depth each
dropout_keep = 0.75 
activation_func = tf.nn.softsign # options = tf.nn.softsign, tf.nn.crelu, tf.nn.relu6, tf.nn.selu, tf.nn.elu, tf.nn.softplus, tf.sigmoid, tf.nn.relu, tf.tanh
# For LDA
n_features = 5000 # The number of features we want to identify for each document
n_components = 5 # To select the total number of topics (clusters) to be identified
n_top_words = 300 # For selecting the top words in each topic (cluster)
# For NMF
nmf_beta_loss = "frobenius" # kullback-leibler (1) or frobenius (2 - default)
solver ='cd' # kullback-leibler (mu) or frobenius (cd - default)
max_iter = 200 # kullback-leibler (1000) or frobenius (200 - default)
log_dir="./runs/rnn_fs/cluster"
dataset = load_dataset(corpus)
''' FEATURE SELECTION '''
# NMF.
#topics_word_list = get_NMF_word_clusters(dataset, n_features, n_components, n_top_words, nmf_beta_loss, solver, max_iter) # For NMF
# LDA
topics_word_list = get_LDA_word_clusters(dataset, n_features, n_components, n_top_words) # For LDA. 2:37 mins
nlp_topics_word_list = [nlp(" ".join(topic)) for topic in topics_word_list]
selected_topics = [i for i in range(0,n_components)] 
''' LOAD DATA - Training and Test'''
print("Loading train-test data...")
t0 = time()
labels, all_data = get_documents(collection, incl_det=incl_det)
print("done in %0.3fs." % (time() - t0))
print("Processing train-test data through SpaCy...", end="", flush=True)
t0 = time()
nlp_all_documents = [nlp(document) for document in all_data] # This takes 31:12 minutes
print("done in %0.3fs." % (time() - t0))
cluster_labels = find_similarities(nlp_all_documents, nlp_topics_word_list) 
print(colored("cluster_labels = ", 'cyan'), end="", flush=True)
print((set(cluster_labels))) # Check how many clusters have actually been used.
reduced_semantic_data = get_semantic_features(all_data, topics_word_list, selected_topics, cluster_labels) 
data = reduced_semantic_data
max_document_length = max([len(i.split(" ")) for i in data]) 
MAX_SEQUENCE_LENGTH = int(round(np.mean([len(i.split(" ")) for i in data]))) 
print(colored('MAX_SEQUENCE_LENGTH = ', 'cyan'), colored(MAX_SEQUENCE_LENGTH, 'yellow', attrs=['bold']))
vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(max_document_length) 
X_vocab = np.array(list(vocab_processor.fit_transform(data))) 
y_output = np.array(labels) 
x_data = X_vocab # No further feature selection required
print(colored('Document size = ', 'cyan'), colored(len(x_data[0]), 'yellow', attrs=['bold']))
vocabulary_size = len(vocab_processor.vocabulary_) # 551
#print('vocabulary_size = ', vocabulary_size)
np.random.seed(22)
shuffle_indices = np.random.permutation(np.arange(len(x_data))) 
x_shuffled = x_data[shuffle_indices]
y_shuffled = y_output[shuffle_indices]
train_data = x_shuffled[:TRAIN_DATA]
train_target = y_shuffled[:TRAIN_DATA]
test_data = x_shuffled[TRAIN_DATA:TOTAL_DATA]
test_target = y_shuffled[TRAIN_DATA:TOTAL_DATA]
## SET UP THE NEURAL NETWORK ##
tf.reset_default_graph()
x = tf.placeholder(tf.int32, [None, len(x_data[0])], name="X") 
y = tf.placeholder(tf.int32, [None], name="labels") 
with tf.name_scope("embedding"):
    embedding_matrix = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0), name="embedMatrix") 
    embeddings = tf.nn.embedding_lookup(embedding_matrix, x) 
#print('Check shape of embedding_matrix: ', embedding_matrix)
#print('Embedding for the current batch of data that will be fed into the NN: ', embeddings)

#cells = [tf.contrib.rnn.BasicRNNCell(num_units=n) for n in num_units]
#cells = [tf.contrib.rnn.GRUCell(num_units=n) for n in num_units]
cells = [tf.contrib.rnn.GRUCell(num_units=n, activation=activation_func) for n in num_units]
cell_type = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)
#cell_type = tf.contrib.rnn.BasicRNNCell(embedding_size) # Instantiate RNN cell.
#cell_type = tf.contrib.rnn.GRUCell(embedding_size) 
print('Cell Type: {}'.format(cell_type.get_config()['name']))
cell_type = tf.contrib.rnn.DropoutWrapper(cell=cell_type, output_keep_prob=dropout_keep) 
print('Wrapper: {}'.format(cell_type.get_config()['name']))
#_, (encoding, _) = tf.nn.dynamic_rnn(cell_type, embeddings, dtype=tf.float32) # Capture state of single LSTM cell.
_, final_states = tf.nn.dynamic_rnn(cell_type, embeddings, dtype=tf.float32) # Build the RNN and capture the output (unimportant), and final states of all GRU cells
encoding = final_states[len(num_units)-1] # Extract the value of the final GRU cell from the MultiRNNCell.
# #### A densely connected prediction layer
with tf.name_scope("loss"):
    logits = tf.layers.dense(encoding, max_label, activation=None) 
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y, name="cross_entropy")
    loss = tf.reduce_mean(cross_entropy, name='mean_loss') 

tf.summary.scalar('loss', loss)

with tf.name_scope("accuracy"):
    prediction = tf.equal(tf.argmax(logits, 1), tf.cast(y, tf.int64)) 
    accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32)) 

tf.summary.scalar('accuracy', accuracy)

with tf.name_scope("train"):
    optimizer = tf.train.AdamOptimizer(0.01) 
    train_step = optimizer.minimize(loss)

merged_summary = tf.summary.merge_all()
saver = tf.train.Saver()
save_path = 'checkpoints/best_validation'
init = tf.global_variables_initializer()
## TRAINING PROCESS ###
with tf.Session() as session: 
    init.run() 
    train_writer = tf.summary.FileWriter(log_dir + '/train', session.graph)
    test_writer = tf.summary.FileWriter(log_dir + '/test')
    writer = tf.summary.FileWriter(log_dir)
    writer.add_graph(session.graph)
    writer.close()
    for epoch in range(num_epochs): 
        num_batches = int(len(train_data) // batch_size) + 1 
        for i in range(num_batches):
            # Select train data
            min_ix = i * batch_size 
            max_ix = np.min([len(train_data), ((i+1) * batch_size)]) 
            x_train_batch = train_data[min_ix:max_ix] 
            y_train_batch = train_target[min_ix:max_ix] 
            train_dict = {x: x_train_batch, y: y_train_batch} 
            session.run(train_step, feed_dict=train_dict)
            train_loss, train_acc = session.run([loss, accuracy], feed_dict=train_dict) 
        test_dict = {x: test_data, y: test_target} 
        #test_loss, test_acc = session.run([loss, accuracy], feed_dict=test_dict) # The loss calcualtions for the Test dataset after every epoch.
        summary, test_acc = session.run([merged_summary, accuracy], feed_dict=test_dict)
        test_writer.add_summary(summary, epoch+1)
        #print('Epoch: {}, Test Loss: {:.2}, Test Acc: {:.5}'.format(epoch + 1, test_acc))
        print('Epoch: {}, Test Acc: {:.5}'.format(epoch + 1, test_acc))
        saver.save(sess=session, save_path=save_path)
    test_writer.close()
    train_writer.close()
    print(colored('Accuracy = ', 'cyan'), colored(test_acc, 'yellow', attrs=['bold']))

save_variables(labels, all_data, cluster_labels, topics_word_list, data, nlp_topics_word_list, vocab_processor)

