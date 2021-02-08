import subprocess
import os
import urllib2
import argparse
from random import randint
from time import sleep

# Note: this uses Python 2
# It should be run as:
# $ python2 <year> <document_number>

def download_file(download_url,output_file):
    try:
        #Wait for a random amount of time between 1 and 5 seconds to avoid auto block
        sleep(randint(1,5))
        response = urllib2.urlopen(download_url)
        file = open(output_file, 'wb')
        file.write(response.read())
        file.close()
        return "success"
    except urllib2.HTTPError, e:
        return "error"

def main(): 
    # Initialisation
    parser = argparse.ArgumentParser(description='Download NZERA determination')
    parser.add_argument('year', type=int, help='Year of the determination')
    parser.add_argument('document', type=int, help='Determination number')
    args = parser.parse_args()
    # old URL sample: http://apps.employment.govt.nz/determinations/PDF/2009/2009_NZERA_AA_1.pdf
    # new URL sample: https://www.employment.govt.nz/assets/elawpdf/2019/2019-NZERA-1.pdf
    urlRoot = "https://www.employment.govt.nz/assets/elawpdf/"
    outputLocationRoot = "./data"

    print args.year, args.document,
    # Create url from the url root, the year and the document number
    url = urlRoot + str(args.year) + "/" + str(args.year) + "-NZERA-" + str(args.document) + ".pdf"
    # Set pdf location and text output location
    outputFile = outputLocationRoot + "/PDF/" + str(args.year) + "-NZERA-" + str(args.document) + ".pdf"
    outputText = outputLocationRoot + "/Text/" + str(args.year) + "-NZERA-" + str(args.document) + ".txt"
    # Download the document and save it in a particular path, Catch error
    attempt = download_file(url,outputFile)
    print "....", attempt
    if attempt == "success":
        # Extract text
        subprocess.call(['pdftotext', outputFile, outputText])
        os.remove(outputFile)
    else:
        # If no document is available, flag this as a document to be extracted the next day
        with open('./var/document', 'w') as f:
            f.write(str(document))

if __name__ == "__main__": 
    main()


