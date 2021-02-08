##This programme runs with the argument $ python ViewInfo.py <file_name>
#e.g. $ python lib/viewInfo.py 2019_1_001
# Predicted label can be manually verified by running:
# $ vim data/2019-NZERA-1.txt
# or by entering the following url into a browser: 
# https://www.employment.govt.nz/assets/elawpdf/2019/2019-NZERA-1.pdf

from pymongo import MongoClient
import os
import sys
from termcolor import colored, cprint

def main():
    client = MongoClient()
    db = client.employmentdb
    collection = db.nzera
    file_name = str(sys.argv[1])
    target_doc = queryMongoDB(collection, file_name)
    outputData(target_doc, collection)

def queryMongoDB(collection, file_name):
    return collection.find_one({"file_name":file_name})
    
def outputData(target_doc, collection):
    os.system('clear')
    do_flag = True
    d_flag = True
    #Print Preamble
    print(colored('DOCUMENT NAME:', 'yellow', attrs=['bold']), colored(target_doc['file_name'], 'cyan'))
    print(colored('PREAMBLE:\n', 'yellow', attrs=['bold']) + colored(target_doc['0'], 'green'))
    print(colored('DESIRED_OUTCOMES:', 'yellow', attrs=['bold']), end="")
    try:
        desired_outcome = target_doc['desired_outcome']
    except KeyError:
        desired_outcome = "No Desired Outcomes present in this case. Check the original document."
        do_flag = False
    print(desired_outcome)
    #Print all Desired_Outcomes
    if do_flag:
        for xyz in target_doc['desired_outcome']:
            print(collection.find_one({"file_name":target_doc['file_name']}, {"_id":0, str(xyz):1})[str(xyz)])
    print(colored('\nDETERMINATIONS:', 'yellow', attrs=['bold']), end="")
    try:
        determinations = target_doc['determinations']
    except KeyError:
        determinations = "No Determinations present in this case. Check the original document."
        d_flag = False
    print(determinations)
    #Print all Determinations
    if d_flag:
        for xyz in target_doc['determinations']:
            print(collection.find_one({"file_name":target_doc['file_name']}, {"_id":0, str(xyz):1})[str(xyz)])

if __name__ == "__main__": 
    main()
