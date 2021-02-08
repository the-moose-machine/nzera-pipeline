from __future__ import print_function
from pymongo import MongoClient
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from termcolor import colored
from tqdm import tqdm
from time import time
import re
import numpy as np
import tensorflow as tf
import spacy
import pickle

def get_documents(collection, key="label", label_flag=True, incl_det = True):
    """ Return all documents as a list, and all labels as another list"""
    documents = []
    labels = []
    TOKEN_REGEX = re.compile("[^$A-Za-z0-9 ]+")
    file_namesCursor = findFile_names(collection, key) # Obtain a list containing names of specific documents
    print("Loading dataset...", end="", flush=True)
    t0 = time()
    for file_name in tqdm(file_namesCursor):
        if label_flag:
            label = extractDocInfo(collection, file_name['file_name'], feature="label")['label'] # Extract a specific feature from the document (in this case, label)
            labels.append(label)
        #determinations = extractDocInfo(collection, file_name['file_name'], feature="determinations")['determinations']
        #desired_outcome = extractDocInfo(collection, file_name['file_name'], feature="desired_outcome")['desired_outcome']
        #merged_paragraphs = set(determinations + desired_outcome) # To avoid duplication of paragraphs. Use if not using entire document
        merged_paragraphs = findAllParagraphs(collection, file_name['file_name'], incl_det) # Only if using entire document (FD)
        #preamble = extractDocInfo(collection, file_name['file_name'], feature="0")['0']
        combined_data = ''
        #for paragraph_no in sorted([str(1000+int(i))[-3:] for i in determinations]): # Use only if determinations alone are being considered (ONLY D)
        for paragraph_no in sorted([str(1000+int(i))[-3:] for i in merged_paragraphs]): # Use only if both determinations and desired_outcomes are being considered (D + DO or FD)
            det_str = str(int(paragraph_no)) # To convert paragraph_no from "001" to "1"
            paragraph_text = extractDocInfo(collection, file_name['file_name'], feature=det_str)[det_str]
            combined_data = combined_data + paragraph_text
        #combined_data = preamble + combined_data # Use only if the preamble is being considered as well, remove if using entire document to avoid duplication (P +)
        combined_data = re.sub(TOKEN_REGEX, '', combined_data).lower()
        documents.append(combined_data)
    print("done in %0.3fs." % (time() - t0))
    return labels, documents

def findFile_names(collection, key='label'):
    """ Return a list of all file_names in the NZERA database on MongoDB """
    return collection.find({key: {"$exists": True}},{"file_name":1, "_id":0}) 

def extractDocInfo(collection, file_name, feature): 
    """ Extract a specific value given a key in the NZERA database on MongoDB"""
    return collection.find_one({"file_name":file_name},{"_id":0, feature:1})

def findAllParagraphs(collection, file_name, incl_det):
    cursor = collection.find_one({"file_name": file_name})
    determinations = cursor['determinations']
    if incl_det:
        all_paras = [i for i in cursor if i.isdigit()]
    else:
        all_paras = [i for i in cursor if i.isdigit() and int(i) not in determinations]
    return all_paras

def get_word_features(model, feature_names, n_top_words):
    '''
    Returns a multidimensional list of clusters (topics) that have been created for a particular model.
    The list is of the shape (n_components, n_top_words)
    '''
    topics_word_list = []
    for topic_idx, topic in enumerate(model.components_):
        lista = [feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]
        topics_word_list.append(lista) 
    return topics_word_list

def select_text_features(topics_word_list, document_words, selected_topics):
    '''Selection of text features maintaining sequential logic.'''
    sequential_features = []
    word_flag = ""
    for word in document_words:
        if not word == word_flag:
            for index, line in enumerate(topics_word_list):
                #print('word = {}\nindex = {}\nselected_topics = {}'.format(word, index, selected_topics))
                if word in topics_word_list[index] and index in selected_topics: # If the word lies in the current cluster and this cluster is selected by the user
                    sequential_features.append(word)
                    word_flag = word
                    break
    return sequential_features

def load_dataset(corpus):    
    # Load the NZERA dataset 
    print("Loading corpus...", end="", flush=True)
    t0 = time()
    with open(corpus) as f: 
        dataset = f.read().splitlines()
    print("done in %0.3fs." % (time() - t0))
    return dataset

def get_LDA_word_clusters(dataset, n_features, n_components, n_top_words):
    print("Extracting tf features for LDA...", end="", flush=True)
    tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=n_features, stop_words='english')
    #tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=n_features) #22.47 seconds (with stop_words)
    t0 = time()
    tf = tf_vectorizer.fit_transform(dataset)
    print("done in %0.3fs." % (time() - t0))
    print("Fitting LDA models with tf features...", end="", flush=True)
    lda = LatentDirichletAllocation(n_components=n_components, max_iter=5, learning_method='online', learning_offset=50., random_state=0) # 169.817 seconds (2:50)
    t0 = time()
    lda.fit(tf)
    print("done in %0.3fs." % (time() - t0))
    #print("\nTopics in LDA model:")
    tf_feature_names = tf_vectorizer.get_feature_names() 
    #print_top_words(lda, tf_feature_names, n_top_words)
    topics_word_list = get_word_features(lda, tf_feature_names, n_top_words)
    return topics_word_list

def get_NMF_word_clusters(dataset, n_features, n_components, n_top_words, nmf_beta_loss='frobenius', slvr='cd', mx_itr=200):
    print("Extracting tf-idf features for NMF...", end="", flush=True)
    tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, max_features=n_features, stop_words='english') # 47.86 seconds (no stopwords)
    #tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, max_features=n_features) # 22.65 seconds (with stop_words).
    t0 = time()
    tfidf = tfidf_vectorizer.fit_transform(dataset)
    print("done in %0.3fs." % (time() - t0))
    print("Fitting the NMF model ({}) with tf-idf features".format(nmf_beta_loss), end="", flush=True)
    t0 = time() 
    nmf = NMF(n_components=n_components, random_state=1, beta_loss=nmf_beta_loss, solver=slvr, max_iter=mx_itr, alpha=.1, l1_ratio=.5).fit(tfidf) #24.43 seconds
    print("done in %0.3fs." % (time() - t0))
    #print("\nTopics in NMF model ({}):".format(nmf_beta_loss))
    tfidf_feature_names = tfidf_vectorizer.get_feature_names()
    topics_word_list = get_word_features(nmf, tfidf_feature_names, n_top_words)
    #print_top_words(nmf, tfidf_feature_names, n_top_words)
    return topics_word_list

def get_semantic_features(all_data, topics_word_list, selected_topics, cluster_labels):
    data = []
    for index, document in enumerate(all_data):
        cluster = cluster_labels[index] # Find the cluster of a particular document
        document_words = document.split() 
        #semantic_features = select_text_features(topics_word_list, document_words, selected_topics) # Obtain the shortened features of the document
        semantic_features = list(set(select_text_features(topics_word_list, document_words, [cluster]))) # Only select words belonging to the cluster
        reduced_document = ' '.join(word for word in semantic_features) 
        data.append(reduced_document) 
    return data

def topic_selection(topics_word_list):
    int_choices = []
    print("Select any one of the following topics (0 - {})".format(len(topics_word_list)-1))
    for topic_idx, topic in enumerate(topics_word_list):
        message = "\nTOPIC #%d: " % topic_idx
        message += " ".join(topics_word_list[topic_idx]) 
        print(message)
    choices = input('\nEnter your choices (eg. 1 2 3  or 1-3 5): ').split()
    int_choices = [int(i) for i in choices if '-' not in i]
    range_choices = [i for i in choices if '-' in i]
    for choice in range_choices:
        u,v = choice.split('-')
        int_choices.extend([i for i in range(int(u),int(v)+1)])
    int_choices.sort()
    return int_choices

def find_similarities(nlp_all_documents, nlp_topics_word_list):
    similarity_labels = []
    for document in nlp_all_documents:
        similarities = [document.similarity(cluster) for cluster in nlp_topics_word_list] # Calculate cosine similarities for each document against all clusters
        similarity_labels.append(similarities.index(max(similarities))) # Select the most similar cluster
    return similarity_labels

def save_variables(labels, all_data, cluster_labels, topics_word_list, data, nlp_topics_word_list, vocab_processor):
    with open ('var/labels', 'wb') as fp:
        pickle.dump(labels, fp)
    with open ('var/all_data', 'wb') as fp:
        pickle.dump(all_data, fp)
    with open ('var/cluster_labels', 'wb') as fp:
        pickle.dump(cluster_labels, fp)
    with open ('var/topics_word_list', 'wb') as fp:
        pickle.dump(topics_word_list, fp)
    with open ('var/data', 'wb') as fp:
        pickle.dump(data, fp)
    with open ('var/nlp_topics_word_list', 'wb') as fp:
        pickle.dump(nlp_topics_word_list, fp)
    with open ('var/vocab_processor', 'wb') as fp:
        pickle.dump(vocab_processor, fp)
    return

def insertFeature(file_name, value, feature, collection, function):
    #function = any one of: ['$set' (to set a single value), '$addToSet' (to add a value to a set)]
    collection.update_one({"file_name":file_name},{function:{feature:value}})
