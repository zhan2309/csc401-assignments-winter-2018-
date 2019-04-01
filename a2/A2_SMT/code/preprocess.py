import re
import os
import string

# test dir for this file
# testDir = "../data/Hansard/Training/"

def nonSeperatedWordHelper(reg):
    # from handout
    nonSeperatedList = ["abord", "accord", "ailleurs", "habitude"]
    if reg.group(1) == "d" and reg.group(3) in nonSeperatedList:
        return reg.group(1) + reg.group(2) + reg.group(3)
    else:
        return reg.group(1) + reg.group(2) + " " + reg.group(3)

def preprocess(in_sentence, language):
    """ 
    This function preprocesses the input text according to language-specific rules.
    Specifically, we separate contractions according to the source language, convert
    all tokens to lower-case, and separate end-of-sentence punctuation 
	
	INPUTS:
	in_sentence : (string) the original sentence to be processed
	language	: (string) either 'e' (English) or 'f' (French)
				   Language of in_sentence
				   
	OUTPUT:
	out_sentence: (string) the modified sentence
    """
    # e.g., preprocess("je t’aime.", "f") should return “SENTSTART je t’ aime . SENTEND”.
    out_sentence = in_sentence
    out_sentence = out_sentence.lower()
    out_sentence = re.sub(r"\n", lambda reg: "", out_sentence)
    # commas, colons and semicolons, parentheses, dashes between parentheses
    out_sentence = re.sub(r"()([,;:\(\)\{\}\[\]])()", lambda reg: reg.group(1) + " " + reg.group(2) + " " + reg.group(3), out_sentence)
    out_sentence = re.sub(r"()(-)()", lambda reg: reg.group(1) + " " + reg.group(2) + " " + reg.group(3), out_sentence)
    # math punctuation
    out_sentence = re.sub(r"()([><+=])()", lambda reg: reg.group(1) + " " + reg.group(2) + " " + reg.group(3), out_sentence)
    if language == "e":
        # quotation marks for english
        out_sentence = re.sub(r"()([\"\'\`])()", lambda reg: reg.group(1) + " " + reg.group(2) + " " + reg.group(3), out_sentence)
    else:
        # dont handle ' now just seperate " and `
        out_sentence = re.sub(r"()([\"\`])()", lambda reg: reg.group(1) + " " + reg.group(2) + " " + reg.group(3), out_sentence)
        # handle '
        out_sentence = re.sub(r"([a-z0-9]+)([\'])(\w+)", nonSeperatedWordHelper, out_sentence)
    out_sentence = "SENTSTART " + out_sentence + " SENTEND"
    # get rid of extra space
    out_sentence = re.sub(r" +", lambda reg: " ", out_sentence)
    out_sentence = re.sub(r"()([\.?!])( SENTEND)", lambda reg: reg.group(1) + " " + reg.group(2) + reg.group(3), out_sentence)
    return out_sentence

# main test
# if __name__ == "__main__":
#     # print("hhhh")
#     for _, _, files in os.walk(testDir):
#         j = 0
#         for file in files:
#             language = ""
#             if file.endswith(".e"):
#                 language = "e"
#             if file.endswith(".f"):
#             	language = "f"
#             if language != "":
#                 openFile = open(testDir+file, "r")
#                 i = 0
#                 for line in openFile.readlines():
#                     print(preprocess(line, language))
#                     i+=1
#                     if i == 10:
#                         break
#             print("--------------------------------------------------------")
#             j +=1
#             if j == 4:
#                 break
#         break

