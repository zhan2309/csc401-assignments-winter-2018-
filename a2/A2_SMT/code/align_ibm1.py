from lm_train import *
from log_prob import *
from preprocess import *
from math import log
import os

def align_ibm1(train_dir, num_sentences, max_iter, fn_AM):
	"""
	Implements the training of IBM-1 word alignment algoirthm. 
	We assume that we are implemented P(foreign|english)
	
	INPUTS:
	train_dir : 	(string) The top-level directory name containing data
					e.g., '/u/cs401/A2_SMT/data/Hansard/Testing/'
	num_sentences : (int) the maximum number of training sentences to consider
	max_iter : 		(int) the maximum number of iterations of the EM algorithm
	fn_AM : 		(string) the location to save the alignment model
	
	OUTPUT:
	AM :			(dictionary) alignment model structure
	
	The dictionary AM is a dictionary of dictionaries where AM['english_word']['foreign_word'] 
	is the computed expectation that the foreign_word is produced by english_word.
	
			LM['house']['maison'] = 0.5
	"""
	AM = {}
	
	# Read training data
	training_data = read_hansard(train_dir, num_sentences)
	
	# Initialize AM uniformly
	AM = initialize(training_data[0], training_data[1])
	
	# Iterate between E and M steps
	i = 0
	while i < max_iter:
		AM = em_step(AM, training_data[0], training_data[1])
		i+=1

	# Save Model
	with open(fn_AM+'.pickle', 'wb') as handle:
		pickle.dump(AM, handle, protocol=pickle.HIGHEST_PROTOCOL)

	return AM
	
# ------------ Support functions --------------
def read_hansard(train_dir, num_sentences):
	"""
	Read up to num_sentences from train_dir.
	
	INPUTS:
	train_dir : 	(string) The top-level directory name containing data
					e.g., '/u/cs401/A2_SMT/data/Hansard/Testing/'
	num_sentences : (int) the maximum number of training sentences to consider
	
	
	Make sure to preprocess!
	Remember that the i^th line in fubar.e corresponds to the i^th line in fubar.f.
	
	Make sure to read the files in an aligned manner.
	"""
	# want to return (eng, fre) by piazza 
	result = {"eng":[], "fre":[]}
	eng_count = 0
	fre_count = 0
	for _, _, files in os.walk(train_dir):
		for file in files:
			language = ""
			if file.endswith(".e"):
				language = "e"
			elif file.endswith(".f"):
				language = "f"
			if language != "":
				openFile = open(train_dir+file, "r")
				for line in openFile.readlines():
					preprocessed_line = preprocess(line, language)
					if (eng_count != num_sentences) and (language == "e"):
						result["eng"].append(preprocessed_line.split())
						eng_count +=1
					if (fre_count != num_sentences) and (language == "f"):
						result["fre"].append(preprocessed_line.split())
						fre_count +=1					
	eng, fre = (result["eng"], result["fre"])
	return eng, fre



def initialize(eng, fre):
	"""
	Initialize alignment model uniformly.
	Only set non-zero probabilities where word pairs appear in corresponding sentences.
	"""
	AM= {}
	AM['SENTSTART'] = {}
	AM['SENTEND'] = {}
	i = 0
	for englishSentence in eng:
		for englishWord in englishSentence:
			if (englishWord != 'SENTSTART') and (englishWord != 'SENTEND'):
				if not englishWord in AM.keys():
					AM[englishWord] = {}
				for frenchWord in fre[i]:
					if (frenchWord != 'SENTEND') and (frenchWord != 'SENTSTART'):
						AM[englishWord][frenchWord] = 1
		i+=1
	
	for word in AM.keys():
		number_pairs = len(AM[word].keys())
		for frenchWord in AM[word].keys():
			AM[word][frenchWord] = 1 / number_pairs
	# from handout
	AM['SENTSTART']['SENTSTART'] = 1
	AM['SENTEND']['SENTEND'] = 1
	return AM
	
def em_step(t, eng, fre):
	"""
	One step in the EM algorithm.
	Follows the pseudo-code given in the tutorial slides.
	"""
	#  set tcount(f, e) to 0 for all f, e
	#  set total(e) to 0 for all e
	#  for each sentence pair (F, E) in training corpus:
	#  	for each unique word f in F:
	#  		denom_c = 0
	#  		for each unique word e in E:
	#  			denom_c += P(f|e) * F.count(f)
	#  		for each unique word e in E:
	#  			tcount(f, e) += P(f|e) * F.count(f) * E.count(e) / denom_c
	#  			total(e) += P(f|e) * F.count(f) * E.count(e) / denom_c
	#  for each e in domain(total(:)):
	#  	for each f in domain(tcount(:,e)):
	#  		P(f|e) = tcount(f, e) / total(e)
	result = {"tcount": {}, "total": {}}
	for i in range(len(eng)):
		englishSentence = eng[i]
		frenchSentence = fre[i]	
		count_total = em_step_helper(englishSentence, frenchSentence)
		count_english = count_total[0]
		count_french = count_total[1]
		for frenchWord in count_french.keys():
			#  		denom_c = 0
			#  		for each unique word e in E:
			#  			denom_c += P(f|e) * F.count(f)
			count_denom = 0
			for englishWord in count_english.keys():
				temp = count_french[frenchWord] * t[englishWord][frenchWord]
				count_denom += temp
			#  for each unique word e in E:
			for englishWord in count_english.keys():
				if englishWord not in result["total"].keys():
					result["total"][englishWord] = 0
				if englishWord not in result["tcount"].keys():
					initializeDic = {}
					result["tcount"][englishWord] = initializeDic
				if frenchWord not in result["tcount"][englishWord].keys():
					result["tcount"][englishWord][frenchWord] = 0
				#  		tcount(f, e) += P(f|e) * F.count(f) * E.count(e) / denom_c
				#  		total(e) += P(f|e) * F.count(f) * E.count(e) / denom_c	
				result["tcount"][englishWord][frenchWord] += t[englishWord][frenchWord] * count_french[frenchWord] * count_english[englishWord] / count_denom							
				result["total"][englishWord] += t[englishWord][frenchWord] * count_french[frenchWord] * count_english[englishWord] / count_denom
	#  for each e in domain(total(:)):
	#  	for each f in domain(tcount(:,e)):
	#  		P(f|e) = tcount(f, e) / total(e)
	for englishWord in result["total"].keys():
		for frenchWord in result["tcount"][englishWord].keys():
			t[englishWord][frenchWord] = result["tcount"][englishWord][frenchWord] / result["total"][englishWord]
	
	return t

def em_step_helper(englishSentence, frenchSentence):
	count_english = {}
	count_french = {}
	for word in englishSentence:
		if (word != 'SENTSTART') and (word != 'SENTEND'):
			if not word in count_english.keys():
				count_english[word] = 1
			else:
				count_english[word] +=1

	for word in frenchSentence:
		if (word != 'SENTSTART') and (word != 'SENTEND'):
			if not word in count_french.keys():
				count_french[word] = 1
			else:
				count_french[word] +=1	
	return count_english , count_french


# test
if __name__ == "__main__":
	testDir = "/u/cs401/A2_SMT/data/Hansard/Testing/"
	engAndFrench = read_hansard(testDir, 5)
	print("read is: ")
	print(engAndFrench)
	am_initial = initialize(engAndFrench[0], engAndFrench[1])
	print("am_initial is:")
	# print(am_initial)
	testDestinationDir = "../data"
	print(align_ibm1(testDir, 10, 1000, testDestinationDir+"AM"))
	