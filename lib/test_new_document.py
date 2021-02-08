from lib.imports_n_methods_2019 import *
from lib.load_variables import *
#import sys

def test_new_document(file_name, incl_det=True):
    # 'Including' determinations may help improve prediction accuracy. Train a model with incl_det=True first.
    client = MongoClient()
    db = client.employmentdb
    collection = db.nzera
    nlp = spacy.load('en_core_web_sm')
    #file_name = '2015_1_090'
    #file_name = '2014_1_458'
    #file_name = str(sys.argv[1])
    #incl_det = False

    merged_paragraphs = findAllParagraphs(collection, file_name, incl_det)
    TOKEN_REGEX = re.compile("[^$A-Za-z0-9 ]+")
    combined_data = ''
    for paragraph_no in sorted([str(1000+int(i))[-3:] for i in merged_paragraphs]):
        det_str = str(int(paragraph_no))
        paragraph_text = extractDocInfo(collection, file_name, feature=det_str)[det_str]
        combined_data = combined_data + paragraph_text

    new_document = re.sub(TOKEN_REGEX, '', combined_data).lower()

    # FEATURE SELECTION
    nlp_new_document = nlp(new_document) # Takes a few seconds
    # Locate the closest cluster
    similarities = [nlp_new_document.similarity(cluster) for cluster in nlp_topics_word_list] # Calculate cosine similarities for the document against all clusters
    new_doc_cluster = similarities.index(max(similarities)) # Select the most similar cluster
    # Reduce the document according to the cluster
    new_document_words = new_document.split()
    semantic_features = list(set(select_text_features(topics_word_list, new_document_words, [new_doc_cluster]))) # Only select words belonging to the cluster
    reduced_new_document = ' '.join(word for word in semantic_features)

    # Adjust vocabulary of the new document
    new_data = [i for i in data]
    #new_data.append(reduced_new_document)
    max_document_length = max([len(i.split(" ")) for i in new_data]) # 281. This does not include the new document
    new_data.append(reduced_new_document)
    #vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(max_document_length) 
    vocab_dict = vocab_processor.vocabulary_._mapping # Extract original vocabulary
    sorted_vocab = sorted(vocab_dict.items(), key = lambda x : x[1]) # Sort the vocabulary
    vocabulary = list(list(zip(*sorted_vocab))[0]) # Extract the vocabulary. Error: Only one word
    extra_words = set(reduced_new_document.split()).difference(vocabulary) # Identify new words in the new doc
    #print('Extra words: ', extra_words)
    print(colored('Extra words in', 'cyan'), colored(file_name, 'cyan'), colored('are : ', 'cyan'), extra_words)
    culled_reduced_new_doc = set(reduced_new_document.split()).difference(extra_words)
    reduced_new_document = ' '.join(word for word in culled_reduced_new_doc)

    # Tokenize
    new_X_vocab = np.array(list(vocab_processor.fit_transform(new_data)))
    new_docX = new_X_vocab[-1:]
    #new_docY = int(collection.find_one({"file_name":file_name}, {"label":1,"_id":0})['label']) # if label exists
    new_docY = 1 # Default outcome = case won. Useful in obtaining prediction_label
    new_docY = np.array(list(str(new_docY)))

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
        predicted_label = session.run(prediction, feed_dict=new_feed_dict)[0] # This is a Boolean

    #print('The predicted label = {}'.format(int(predicted_label)))
    print(colored('The predicted label for', 'cyan'), colored(file_name, 'cyan'), colored('= ', 'cyan'), colored(int(predicted_label), 'yellow', attrs=['bold']))
    return int(predicted_label)

    # Save int(predicted_label) to the 'label' field in Mongo

