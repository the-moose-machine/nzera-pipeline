from lib.imports_n_methods_2019 import *
import pickle

client = MongoClient()
db = client.employmentdb
collection = db.nzera
nlp = spacy.load('en_core_web_sm')
corpus = "/home/moose/Documents/Education/allText.txt" 
incl_det = False 
labels, all_data = get_documents(collection, incl_det=incl_det)

file_name = '2015_1_090' # This needs to be adjusted to grab the name of the new document
# Locate all paragraphs of a new document: 2015_1_090
merged_paragraphs = findAllParagraphs(collection, file_name, incl_det)
# Extract content of relevant paragraphs
TOKEN_REGEX = re.compile("[^$A-Za-z0-9 ]+")
combined_data = ''
for paragraph_no in sorted([str(1000+int(i))[-3:] for i in merged_paragraphs]):
    det_str = str(int(paragraph_no))
    paragraph_text = extractDocInfo(collection, file_name, feature=det_str)[det_str]
    combined_data = combined_data + paragraph_text

new_document = re.sub(TOKEN_REGEX, '', combined_data).lower()

# Load Data, NLP
# For LDA
n_features = 5000 
n_components = 5 
n_top_words = 300
dataset = load_dataset(corpus)
topics_word_list = get_LDA_word_clusters(dataset, n_features, n_components, n_top_words) # For LDA. 2:37 mins
nlp_topics_word_list = [nlp(" ".join(topic)) for topic in topics_word_list]

# FEATURE SELECTION
nlp_new_document = nlp(new_document) # Takes a few seconds
# Locate the closest cluster
similarities = [nlp_new_document.similarity(cluster) for cluster in nlp_topics_word_list] # Calculate cosine similarities for the document against all clusters
new_doc_cluster = similarities.index(max(similarities)) # Select the most similar cluster
# Reduce the document according to the cluster
new_document_words = new_document.split()
semantic_features = list(set(select_text_features(topics_word_list, new_document_words, [new_doc_cluster]))) # Only select words belonging to the cluster
reduced_new_document = ' '.join(word for word in semantic_features)

# TEST DOCUMENT AGAINST MODEL
# Tokenization
n_components = 5 
selected_topics = [i for i in range(0,n_components)] 
cluster_labels = [0, 3, 4, 4, 4, 4, 0, 4, 4, 4, 0, 3, 3, 0, 0, 4, 0, 0, 3, 0, 3, 0, 4, 3, 4, 4, 4, 3, 0, 4, 4, 3, 4, 0, 3, 4, 3, 4, 0, 4, 4, 4, 0, 0, 0, 0, 4, 0, 0, 3, 0, 4, 4, 0, 0, 0, 4, 4, 0, 4, 4, 4, 0, 4, 0, 0, 0, 4, 0, 3, 4, 4, 4, 4, 3, 0, 4, 0, 0, 0, 4, 4, 0, 3, 4, 4, 3, 0, 4, 4, 4, 4, 4, 0, 4, 0, 4, 4, 4, 4, 0, 4, 3, 0, 4, 4, 0, 4, 0, 3, 0, 4, 4, 4, 0, 4, 4, 4, 0, 4, 4, 4, 0, 4, 3, 4, 4, 4, 4, 0, 0, 4, 4, 4, 0, 4, 3, 4, 0, 4, 3, 4, 0, 0, 4, 4, 3, 3, 4, 3, 4, 4, 4, 0, 0, 0, 0, 0, 4, 4, 4, 3, 0, 4, 0, 0, 4, 0, 0, 0, 0, 3, 4, 4, 0, 4, 0, 4, 0, 4, 4, 4, 0, 4, 0, 4, 4, 0, 4, 4, 4, 3, 4, 3, 4, 0, 0, 4, 4, 0, 3, 0, 0, 0, 0, 4, 4, 0, 3, 0, 0, 0, 4, 4, 4, 0, 0, 0] # For testing purposes
data = get_semantic_features(all_data, topics_word_list, selected_topics, cluster_labels) # This does not include the new document

#vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(281)
new_data = [i for i in data]
new_data.append(reduced_new_document)
max_document_length = max([len(i.split(" ")) for i in new_data]) # This does not include the new document
vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(max_document_length) 
new_X_vocab = np.array(list(vocab_processor.fit_transform(new_data)))
new_docX = new_X_vocab[-1:]
#new_docY = int(collection.find_one({"file_name":file_name}, {"label":1,"_id":0})['label']) # if label exists
new_docY = 1 # Default outcome = case won. Useful in obtaining prediction_label
new_docY = np.array(list(str(new_docY)))
#new_feed_dict = {x: new_docX, y: new_docY}

# Set up NN
embedding_size = 128 
no_cells = 1
#vocabulary_size =  551
vocabulary_size = len(vocab_processor.vocabulary_)
print('Vocabulary Size = {}'.format(vocabulary_size))
dropout_keep = 0.75 
max_label = 2 
num_units = [embedding_size] * no_cells
activation_func = tf.nn.softsign
tf.reset_default_graph()
#x = tf.placeholder(tf.int32, [None, 281], name="X") 
x = tf.placeholder(tf.int32, [None, len(new_X_vocab[0])], name="X") 
y = tf.placeholder(tf.int32, [None], name="labels") 
new_feed_dict = {x: new_docX, y: new_docY}
with tf.name_scope("embedding"):
    embedding_matrix = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0), name="embedMatrix") 
    embeddings = tf.nn.embedding_lookup(embedding_matrix, x) 

cells = [tf.contrib.rnn.GRUCell(num_units=n, activation=activation_func) for n in num_units]
cell_type = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)
cell_type = tf.contrib.rnn.DropoutWrapper(cell=cell_type, output_keep_prob=dropout_keep) 
_, final_states = tf.nn.dynamic_rnn(cell_type, embeddings, dtype=tf.float32) # Is this required?
encoding = final_states[len(num_units)-1]
with tf.name_scope("loss"):
    logits = tf.layers.dense(encoding, max_label, activation=None) 
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y)
    loss = tf.reduce_mean(cross_entropy) 

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
# Load session and test for new value
with tf.Session() as session: 
    #init.run() 
    saver.restore(sess=session, save_path=save_path)
    # Feed this processed document into the model and extract the output
    #summary, test_acc = session.run([merged_summary, accuracy], feed_dict=new_feed_dict)
    #print('Test Accuracy = {}'.format(test_acc))
    # Extract the label only
    # To do this, I think that the label value may need to be programmed first.
    predicted_label = session.run(prediction, feed_dict=new_feed_dict)[0] # This is a Boolean

print('The predicted label = {}'.format(int(predicted_label)))

    # Test this without feeding the label and acquire the predicted label only.

# Test with new untested document

# Test with direct pipeline
