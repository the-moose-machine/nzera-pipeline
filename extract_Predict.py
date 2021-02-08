#!/usr/bin/python

import subprocess
import datetime
from lib.extractParagraphs import *
from lib.extractFeature import *
from lib.test_new_document import *

def main(): 
    now = datetime.datetime.now()
    year = now.year
    with open('./var/document', 'r') as f:
        document = int(f.read())
    #Start a loop to extract all new documents starting from the saved flag and incrementing by 1
    subprocess.call(['python2','./lib/crawler.py', str(year), str(document)])
    # Location of extracted text file
    orig_dir = '../data/Text/'
    fn = str(year)+ "-NZERA-" + str(document) + ".txt"
    
    # Save document number
    with open('./var/document', 'w') as f:
        document += 1
        f.write(str(document))
    # Extract paragraphs, save document into Mongo and obtain the file_name key for Mongo
    formatted_fn, collection = extractParagraphs(fn)
    print("file_name: {}".format(formatted_fn))
    # Feature extraction: determinations and desired outcomes
    det_flag = extractFeature(collection, formatted_fn, feature="determinations")
    _ = extractFeature(collection, formatted_fn, feature="desired_outcome")
    
    # Predict label, if determinations exist. 
    # Perhaps the 'determinations' clause could be removed in the future
    if det_flag:
        predicted_label = test_new_document(formatted_fn)
        # Store predicted label into Mongo
        feature="label"
        function="$set"
        insertFeature(formatted_fn, str(predicted_label), feature, collection, function)
        print("predicted_label ({}) added to MongoDB. Please verify accuracy by running:".format(str(predicted_label)))
        print("$ python lib/viewInfo.py {}".format(formatted_fn))
        # If inaccurate update Mongo manually with: 
        # >>> collection.update_one({"file_name":formatted_fn},{"$set":{"label":str(corrected_label)}})
        # where, corrected_label = 0 or 1
    

if __name__ == "__main__": 
    main()
