import pickle
#from lib.imports_n_methods_2019 import *

with open ('var/labels', 'rb') as fp:
    labels = pickle.load(fp)

with open ('var/all_data', 'rb') as fp:
    all_data = pickle.load(fp)

with open ('var/cluster_labels', 'rb') as fp:
    cluster_labels = pickle.load(fp)

with open ('var/topics_word_list', 'rb') as fp:
    topics_word_list = pickle.load(fp)

with open ('var/data', 'rb') as fp:
    data = pickle.load(fp)

with open ('var/nlp_topics_word_list', 'rb') as fp:
    nlp_topics_word_list = pickle.load(fp)

with open ('var/vocab_processor', 'rb') as fp:
    vocab_processor = pickle.load(fp)

selected_topics = [i for i in range(0,5)]
