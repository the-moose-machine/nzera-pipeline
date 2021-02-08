from pymongo import MongoClient
import re
import pickle

def extractFeature(collection, file_name, feature="determinations"):
    #inclusive_list = ['I award', 'I also award'] ### CHECK THIS ###
    #exclusive_list = []
    
    if feature == "determinations":
        with open ('var/d_terms', 'rb') as fp:
            terms = pickle.load(fp)
    elif feature == "desired_outcome":
        with open ('var/do_terms', 'rb') as fp:
            terms = pickle.load(fp) 
    else:
        print("Incorrect feature passed for extraction")
        return
    
    #for index, term in enumerate(terms):
        #inclusive_list = term[index][0]
        #exclusive_list = term[index][1]
    for term in terms:
        inclusive_list = term[0]
        exclusive_list = term[1]
    
        #Find the relevant file
        doc_dict = collection.find_one({"file_name":file_name})
        file_length = len(doc_dict)-2 #Since the current structure has 2 extra fields
        features_identified = set()
        
        if file_length < 1:
            print("{} has insufficient fields".format(file_name))
            continue
        
        for paragraph in range(1,file_length):
            try:
                extracted_para = doc_dict[str(paragraph)] # Extract paragraph text
            except:
                pass
            else:
                comparison_result = find_strings(inclusive_list, exclusive_list, extracted_para)
                
                #if extracted_string in extracted_para: # For direct search
                if comparison_result == True: # For Boolean search
                    features_identified.add(paragraph)
            insertDetermination(file_name, features_identified, feature, collection)
    try:
        # Return boolean value to indicate the existence of the selected feature within this document
        flag = bool(collection.find_one({"file_name":file_name},{feature:1, "_id":0})[feature])
    except KeyError:
        flag = False
    return flag
        
def insertDetermination(fileName, features_identified, feature, collection):
    for x in features_identified:
        collection.update_one({"file_name":fileName},{"$addToSet":{feature:x}})


def find_strings(incl_list, excl_list, extParagraph):
    for string in incl_list:
        if string in extParagraph:
            for item in excl_list:
                if item in extParagraph:
                    return False
            return True
    return False

if __name__ == "__main__":
    client = MongoClient()
    #db = client.test_database # For testing purposes
    db = client.employmentdb # For deployment
    collection = db.nzera
    feature = "determinations"
    extractFeature(collection, file_name, feature)
