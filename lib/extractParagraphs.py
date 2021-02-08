import sys
import re
from pymongo import MongoClient

# Run this script as follows:
# $ python extractDeterminations.py 2014_1_458

def extractParagraphs(fn):
    #MAIN
    orig_dir = './data/Text/'
    #global paragraphs
    #global collection
    client = MongoClient()
    #db = client.test_database # use a database called "test_database"
    #db = client.test_db_nonstd # test database for non-std documents
    db = client.employmentdb # use a database called "employmentdb"
    collection = db.nzera # and inside that DB, a collection called "nzera"
    
    paraIndex = 0
    paragraphs = [""]
    
    # Read all lines
    with open(orig_dir + fn) as f:
        lines = f.readlines()
    
    #Read one line
    #print("file_name = {}".format(fn))
    for line in lines:
        
        #If the line is '\n' and add them to the current paragraph and move on to the next one
        if line == "\n":
            appendParagraph(paraIndex, line, paragraphs)
            continue
        
        #Extract the first integer in this line
        try:
            testInt = int(re.findall('\d+',line)[0])
        except IndexError:
            testInt = 9999 # Random Number
        
        #Does this line start with '[' and not with a year - FOR STD FORMAT
        if line[0] == "[" and testInt < 1900:
        
        #Does this line start with a number (not a year) followed by a '.' - FOR A NON-STD FORMAT
        #print(.format(testInt, line, line[len(str(testInt))], len(str(testInt))))
        #print("testInt = {0}\nlen(str(testInt = {4}\nline =\n{1}\nline[:len(str(testInt))] = {3}\nline[len(str(testInt))] = {2}\n-*-*-*-*-\n".format(testInt, line, line[len(str(testInt))], line[:len(str(testInt))],len(str(testInt))))
        #if len(line) > len(str(testInt)):
            #if line[len(str(testInt))] == "." and testInt < 1900 and testInt != 9999 and line[:len(str(testInt))] == str(testInt):
            #if line[len(str(testInt))] == "." and testInt < 1900 and line[:len(str(testInt))] == str(testInt):
                
            #If yes, then this is a new paragraph
            #Fist save the paragraph in MongoDB for this particular document ##### TO BE DONE ######
            
            #Then, set new index for the list of paragraphs
            paraIndex += 1
                
            #Append new list item with a blank
            paragraphs.append("")
        #If no, then this is the same paragraph
            #No change to index
        
        #If the sentence does not start with a page break (form feed):
        if line[0] != "\f":
            #Append current line to the currently indexed item (index, line) in the list of paragraphs
            appendParagraph(paraIndex, line, paragraphs)
  
    #Output a list of files that are not in the standardised format - DONE
    #outputNonStandardList(fn, paragraphs)
    
    #Delete all [nums] from the start of each paragraph.
    #delete_nums()
    
    #First locate the correct entry name in YYYY_C_DDD format
    formatted_fn = convertFileName(orig_dir, fn)
    #Load all the paragraphs from this file into MongoDB
    save_to_Mongo(formatted_fn, collection, paragraphs) # Is this actually saving?
    return formatted_fn, collection


def appendParagraph(pInd,sentence, paragraphs):
    #SUBROUTINE: Append line
    #Extract the currently indexed item
    currPara = paragraphs[pInd]
    #If the entire string begins with a '[', then remove line breaks (so that the rest of it is not affected), and also remove the numbers within the square brackets #### TO BE DONE ####
    if sentence[-1] == "\n" and pInd > 0:
        sentence = sentence[:-1]
    #Add the contents of the current line to the extracted content
    currPara = currPara + " " + sentence
    #Replace the currently indexed item in the list with the new string
    paragraphs[pInd] = currPara

def outputNonStandardList(fileName, paragraphs):
    try:
        #print("FILE NAME: {}\nFIRST PARAGRAPH:\n{}\nLAST PARAGRAPH:\n{}".format(fileName,paragraphs[1],paragraphs[-1]))
        paragraphs[1] != ""
    except IndexError:
        #print("FILE NAME: {} IS NOT IN A STANDARDISED FORMAT".format(fileName))
        #continue
        nonStd = open('/home/moose/Dropbox/Academics/NLP/ETL/Transform/NonStandardisedDocuments','a')
        nonStd.write(fileName + '\n')
        nonStd.close()

def delete_nums():
    #Delete the [NUM] from the start of each paragraph, apart from the one at index 0
    return

def convertFileName(orig_dir, fileName):
    #Convert original file name into desired format: YYYY_C_DDD
    converted_fn = ""
    ##Calculate YYYY
    YYYY = fileName[:4]
    ##Calculate C
    # IN THE NEW FORMAT C IS IN THE SECOND LINE OF THE TEXT FILE
    # THIS WORKS:
    with open(orig_dir + fileName) as f:
        lines = f.readlines()
    city = lines[1][:-1] # AUCKLAND, CHRISTCHURCH, WELLINGTON
    # Here city[0] = A, C or W
    if city.lower() == "auckland":
         C = "1"
    elif city.lower() == "christchurch":
        C = "2"
    elif city.lower() == "wellington":
        C = "3"
    else:
        print("Naming error for file:" + fn)
    #print("fn[11:12] = " + fn[11:12] + " and C = " + C)
    ##Calculate DDD
    d_temp = abs(int(re.sub(r'[^\d-]+', '', fileName[-7:-4])))
    d_string = "00" + str(d_temp)
    DDD = d_string[-3:]
    ## Compile cleaned_file name
    converted_fn = YYYY + "_" + C + "_" + DDD
    return converted_fn

def save_to_Mongo(fileName, collection, paragraphs):
    #create all key names in the format "XXX", where X is the paragraph number (index number)
    #print("saving {} to {}".format(fileName, collection))
    for index,line in enumerate(paragraphs):
        #print("index = {0}\nline={1}".format(index, line))
        #Append the entry to YYYY_C_DDD with each paragraph (e.g. key: "1", value:"[1] This is...")
        if index == 0:
            collection.insert_one({"file_name":fileName})
        collection.update_one({"file_name":fileName},{'$set':{str(index):line}})
        #print(collection.find().count())
    #print("Total number of indices = {}".format(index))
    #print("Total number of documents = {}\n".format(collection.find().count()))
    #print("Converted file name :{}".format(fileName))
    #cursor = collection.find({"file_name":fileName},{"file_name":1,"_id":0})
    #for output in cursor:
        #print(output)
    return

if __name__ == "__main__": 
    fn = str(sys.argv[1])
    extractDeterminations(fn)
