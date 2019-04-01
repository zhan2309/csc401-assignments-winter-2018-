from preprocess import *
from lm_train import *
from math import log

# testDir = "../data/Hansard/Training/"

def log_prob(sentence, LM, smoothing=False, delta=0, vocabSize=0):
	"""
	Compute the LOG probability of a sentence, given a language model and whether or not to
	apply add-delta smoothing
	
	INPUTS:
	sentence :	(string) The PROCESSED sentence whose probability we wish to compute
	LM :		(dictionary) The LM structure (not the filename)
	smoothing : (boolean) True for add-delta smoothing, False for no smoothing
	delta : 	(float) smoothing parameter where 0<delta<=1
	vocabSize :	(int) the number of words in the vocabulary
	
	OUTPUT:
	log_prob :	(float) log probability of sentence
	"""
	log_prob = 0
	denominator = 0
	numerator = 0
	word_list = sentence.split()
	length_w = len(word_list) -1
	for i in range(length_w):
		word = word_list[i]
		if word in LM['uni'].keys():
			denominator = LM['uni'][word]
		else:
			denominator = 0
		next_word = word_list[i+1]
		if word in LM['bi'].keys() and next_word in LM['bi'][word].keys():
			numerator = LM['bi'][word][next_word]
		else:
			numerator = 0
		# case when numerator/denominator = 0/0 and if it is not smoothing the log_prob will be -infinite
		if (numerator == 0 or denominator == 0) and (not smoothing):
			return float('-inf')
		else:
			numerator += delta
			denominator += delta*vocabSize
			prob = (numerator) / (denominator)
			log_prob+= log(prob, 2)
	return log_prob

# test
if __name__ == "__main__":
	testDir = "/u/cs401/A2_SMT/data/Hansard/Testing/"
	testDestinationDir = "../data"
	test_LM = lm_train(testDir, "e", testDestinationDir + "/English")
	sentence = "SENTSTART canada post . SENTEND"
	sentence_1 = "SENTSTART we live in a democracy . SENTEND"
	log_p = log_prob(sentence, test_LM)
	print(log_p)
	log_p_1 = log_prob(sentence_1, test_LM)
	print(log_p_1)