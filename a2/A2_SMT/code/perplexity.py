from log_prob import *
from preprocess import *
import os

def preplexity(LM, test_dir, language, smoothing = False, delta = 0):
    """
	Computes the preplexity of language model given a test corpus
	
	INPUT:
	
	LM : 		(dictionary) the language model trained by lm_train
	test_dir : 	(string) The top-level directory name containing data
				e.g., '/u/cs401/A2_SMT/data/Hansard/Testing/'
	language : `(string) either 'e' (English) or 'f' (French)
	smoothing : (boolean) True for add-delta smoothing, False for no smoothing
	delta : 	(float) smoothing parameter where 0<delta<=1
	"""
	
    files = os.listdir(test_dir)
    pp = 0
    N = 0
    vocab_size = len(LM["uni"])
    
    for ffile in files:
        if ffile.split(".")[-1] != language:
            continue
        
        opened_file = open(test_dir+ffile, "r")
        for line in opened_file:
            processed_line = preprocess(line, language)
            tpp = log_prob(processed_line, LM, smoothing, delta, vocab_size)
            
            if tpp > float("-inf"):
                pp = pp + tpp
                N += len(processed_line.split())
        opened_file.close()
    if N > 0:
        pp = 2**(-pp/N)
    return pp

# #test
# test_LM = lm_train("lm_train_testdir/", "e", "e_temp")
# print(preplexity(test_LM, "lm_train_testdir/", "e"))
# if __name__ == "__main__":
#     testDir = "../data/Hansard/Training/"
#     testDestinationDir = "../data"
#     test_LM = lm_train(testDir, "e", testDestinationDir + "/English")
#     preplexity_non_smoothing = preplexity(test_LM, testDir, "e")
#     print(preplexity_non_smoothing)
#     preplexity_smoothing_025 = preplexity(test_LM, testDir, "e", True, 0.25)
#     print(preplexity_smoothing_025)
#     preplexity_smoothing_05 = preplexity(test_LM, testDir, "e", True, 0.5)
#     print(preplexity_smoothing_05)
#     preplexity_smoothing_075 = preplexity(test_LM, testDir, "e", True, 0.75)
#     print(preplexity_smoothing_075)
#     preplexity_smoothing_0001 = preplexity(test_LM, testDir, "e", True, 0.001)
#     print(preplexity_smoothing_0001)