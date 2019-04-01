from preprocess import *
import pickle
import os

# test dir for this file
# /u/cs401/A2_SMT/data/Hansard/Training/
# testDir = "../data/Hansard/Training/"
# testDestinationDir = "../data"


def lm_train(data_dir, language, fn_LM):
    """
        This function reads data from data_dir, computes unigram and bigram counts,
        and writes the result to fn_LM

        INPUTS:

    data_dir	: (string) The top-level directory continaing the data from which
                                        to train or decode. e.g., '/u/cs401/A2_SMT/data/Toy/'
        language	: (string) either 'e' (English) or 'f' (French)
        fn_LM		: (string) the location to save the language model once trained

    OUTPUT

        LM			: (dictionary) a specialized language model

        The file fn_LM must contain the data structured called "LM", which is a dictionary
        having two fields: 'uni' and 'bi', each of which holds sub-structures which 
        incorporate unigram or bigram counts

        e.g., LM['uni']['word'] = 5 		# The word 'word' appears 5 times
                  LM['bi']['word']['bird'] = 2 	# The bigram 'word bird' appears 2 times.
    """
    LM = {}
    LM["uni"] = {}
    LM["bi"] = {}
    for _, _, files in os.walk(data_dir):
            for file in files:
                language_ = ""
                if language == "e":
                    language_ = ".e"
                else:
                    language_ = ".f"
                if file.endswith(language_):
                    openFile = open(data_dir+file, "r")
                    for line in openFile.readlines():
                        preprocessedLine = preprocess(line, language)
                        # print(preprocessedLine)
                        word_list = preprocessedLine.split()

                        # Set up LM["uni"]
                        for word in word_list:
                            if word in LM["uni"].keys():
                                LM["uni"][word] += 1
                            else:
                                LM["uni"][word] = 1
                        # Set up LM["bi"]
                        length_w = len(word_list) - 1
                        for index in range(length_w):
                            word_1 = word_list[index]
                            word_2 = word_list[index + 1]
                            # if first word does appears in LM["bi"] then we create first word 
                            # to the LM["bi"] and the second word doesn't have value as well we need give it value 1
                            if word_1 not in LM["bi"].keys():
                                LM["bi"][word_1] = {word_2: 1}
                            else:
                                # if the first word has appeared in LM["bi"] dic then we should check if the second 
                                # word exsits inside the first word dic. if the second word exists, then we simply add 
                                # one else create this word with initial value 1
                                if word_2 not in LM["bi"][word_1].keys():
                                    LM["bi"][word_1][word_2] = 1
                                else:
                                    LM["bi"][word_1][word_2] += 1
                                

    # Save Model
    with open(fn_LM+'.pickle', 'wb') as handle:
        pickle.dump(LM, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return LM

if __name__ == "__main__":
    lm_train("/u/cs401/A2_SMT/data/Hansard/Testing/", "e", testDestinationDir + "/English")
    lm_train("/u/cs401/A2_SMT/data/Hansard/Testing/", "f", testDestinationDir + "/French")
