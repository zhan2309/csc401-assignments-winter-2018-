# run python3 a1_preproc.py 1002137957 -o preproc.json
import sys
import argparse
import os
import json
import html
import re
import string
import spacy

# indir = '/u/cs401/A1/data/';
indir = '/u/cs401/A1/data'
wordlist_dir = '/u/cs401/wordlists/'
abbrList = []
with open(wordlist_dir + "abbrev.english") as abbFile:
    abbrList = abbFile.read().lower().splitlines()
stopWordList = []
with open(wordlist_dir + "StopWords") as abbFile:
    stopWordList = abbFile.read().lower().splitlines()


nlp = spacy.load('en', disable=['parser', 'ner'])

def preproc1( comment , steps=range(1,11)):
    ''' This function pre-processes a single comment

    Parameters:                                                                      
        comment : string, the body of a comment
        steps   : list of ints, each entry in this list corresponds to a preprocessing step  

    Returns:
        modComm : string, the modified comment 
    '''

    modComm = ''
    if 1 in steps:
        #1. Remove all newline characters
        modComm = comment.replace('\n', ' ')
        # print(modComm)
    if 2 in steps:
        #2. Replace HTML character codes with their ASCII equivalent
        modComm = html.unescape(modComm)
        # print(modComm)
    if 3 in steps:
        #3. Remove all URLs (i.e., tokens beginning with http or www).
        modComm = re.sub(r"http\S+", lambda reg: " ", modComm)
        modComm = re.sub(r"www\S+", lambda reg: " ", modComm)
        # print(modComm)
    if 4 in steps:
        #Apostrophes.
        # • Periods in abbreviations (e.g., e.g.) are not split from their tokens. E.g., e.g. stays e.g.
        # • Multiple punctuation (e.g., !?!, ...) are not split internally. E.g., Hi!!! becomes Hi !!!
        # • You can handle single hyphens (-) between words as you please. E.g., you can split non-committal
        # into three tokens or leave it as one.````
        modComm = add_space(modComm)
        #print(modComm)
        
    if 5 in steps:
        # Clitics are contracted forms of words, such as n’t, that are concatenated with the previous word.
        # • Note: the possessive ’s has its own tag and is distinct from the clitic ’s, but nonetheless must
        # be separated by a space; likewise, the possessive on plurals must be separated (e.g., dogs ’).
        temp = splitCliticsHelper(modComm)
        modComm = temp
        # print(modComm)
    if 6 in steps:
        #         A tagged token consists of a word, the ‘/’ symbol, and the tag (e.g., dog/NN). See below for
        # information on how to use the tagging module. The tagger can make mistakes.
        temp = addTagHelper(modComm)
        modComm = temp
        #print(modComm)
    if 7 in steps:
        temp = removeStopWordsHelper(modComm)
        modComm = temp
        #print(modComm)
    if 8 in steps:
        temp = lemmatizationHelper(modComm)
        modComm = temp
        # print(modComm)
    if 9 in steps:
        temp = addNewLineHelper(modComm)
        modComm = temp
        # print(modComm)
    if 10 in steps:
        temp = lowerCaseHelper(modComm)
        modComm = temp
        # print(modComm)

    return modComm

def splitCliticsHelper(modComm):
    modComm = re.sub("(\w)(\w'[^s]\s|s'|'s|'\w+)",lambda reg: reg.group(1) + " " +reg.group(2),modComm)
    modComm = re.sub(' +', ' ', modComm)
    return modComm

def addTagHelper(modComm):
    modComm = re.sub(' +', ' ', modComm)
    temp = modComm.split()
    comm = ""
    i = 0
    while i < len(temp):
        tokenMesg = nlp(temp[i])
        comm += temp[i] + "/" + tokenMesg[0].tag_ + " "
        i+= 1
    modComm = comm
    return modComm

def removeStopWordsHelper(modComm):
    comm = ""
    temp = modComm.split()
    i = 0
    while i < len(temp):
        word = temp[i]
        splited_word = word.split("/")
        if not splited_word[0] in stopWordList:
            comm = comm + " " + word
        i+= 1
    modComm = comm[1:]
    return modComm
    
def lemmatizationHelper(modComm):
    comm = ""
    comm_result = ""
    temp = modComm.split()
    i = 0
    while i < len(temp):
        comm += " " + temp[i].split("/")[0]
        i+=1
    comm = comm[1:]
    doc = spacy.tokens.Doc(nlp.vocab, words=comm.split())
    tokens = nlp.tagger(doc)
    for token in tokens:
        firstChar = token.lemma_[0]
        if firstChar == '-':
            comm_result += " " + token.text +"/"+ token.tag_
        else:
            comm_result += " " + token.lemma_ +"/"+ token.tag_
    modComm = comm_result[1:]
    return modComm

def addNewLineHelper(modComm):
    comm = []
    temp = modComm.split()
    i = 0
    while i < len(temp):
        word = temp[i].split("/")[0]
        tag = temp[i].split("/")[1]
        if word == ".":
            comm.append(temp[i] + '\n')
        elif tag == "." and (word not in abbrList):
            comm.append(temp[i] + '\n')
        else:
            comm.append(temp[i])
        i+=1
    modComm = " ".join(comm)
    modComm = modComm.rstrip('\n')
    abc = modComm.split("\n")
    j = 1
    while j < len(abc):
        abc[j] = abc[j][1:]
        j+=1
    k = 0
    modComm = ""
    while k < len(abc):
        abc[k] = abc[k]+ '\n'
        modComm+= abc[k]
        k += 1
    modComm = modComm.rstrip('\n')
    return modComm

def lowerCaseHelper(modComm):
    temp = re.sub('([\d\w]+)(\/)([A-Za-z\.?:!@,#$%&~;`\(\)\'"]+)', lambda reg: reg.group(1).lower()+ reg.group(2) + reg.group(3), modComm)
    return temp

def add_space(comm):
    comm = re.sub('([\d\w]+)(\.[^\d\w]*)', lambda reg: reg.group(1) +" " +reg.group(2) + " ", comm)             
    comm = re.sub('([\d\w]+)([,;@#$%&:?!\]\."]+[^\d\w]*)', lambda reg: reg.group(1) +" " +reg.group(2) + " ", comm) 
    comm = re.sub(' +', ' ', comm)
    comm = re.sub('\[', '[ ', comm)
    return comm

def main( args ):

    allOutput = []
    for subdir, dirs, files in os.walk(indir):
        for file in files:
            fullFile = os.path.join(subdir, file)
            print( "Processing " + fullFile)

            data = json.load(open(fullFile))

            # TODO: select appropriate args.max lines
            length = args.max
            myId = args.ID[0]
            startIndex = myId % len(data)
            i = startIndex;
            while i < args.max + startIndex:
                data_i = data[i]
                # TODO: read those lines with something like `j = json.loads(line)`
                # TODO: choose to retain fields from those lines that are relevant to you
                # TODO: add a field to each selected line called 'cat' with the value of 'file' (e.g., 'Alt', 'Right', ...) 
                # TODO: process the body field (j['body']) with preproc1(...) using default for `steps` argument
                # TODO: replace the 'body' field with the processed text
                # TODO: append the result to 'allOutput'
                j = json.loads(data_i)
                if not (j['body'] == None) or not (j['body'] == ""):
                    opt_obj = {"id": j["id"], "body": preproc1(j['body']), "cat": file}
                    # print(opt_obj)
                    allOutput.append(opt_obj)
                i+=1


            
    fout = open(args.output, 'w')
    fout.write(json.dumps(allOutput))
    fout.close()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process each .')
    parser.add_argument('ID', metavar='N', type=int, nargs=1,
                        help='your student ID')
    parser.add_argument("-o", "--output", help="Directs the output to a filename of your choice", required=True)
    parser.add_argument("--max", help="The maximum number of comments to read from each file", default=10000)
    args = parser.parse_args()

    if (args.max > 200272):
        print( "Error: If you want to read more than 200,272 comments per file, you have to read them all." )
        sys.exit(1)
        
    main(args)
