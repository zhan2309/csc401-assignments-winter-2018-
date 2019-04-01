# run python3 a1_extractFeatures.py -i preproc.json -o out.json
import numpy as np
import sys
import argparse
import os
import json
import string
import re
import csv

# indir = '/u/cs401/A1/data/';
indir = '../data'
# source list
bristol = "/u/cs401/wordlists/BristolNorms+GilhoolyLogie.csv"
ratings_warriner = "/u/cs401/wordlists/Ratings_Warriner_et_al.csv"
alt = "/u/cs401/A1/feats/Alt_IDs.txt"
left = "/u/cs401/A1/feats/Left_IDs.txt"
right = "/u/cs401/A1/feats/Right_IDs.txt"
center = "/u/cs401/A1/feats/Center_IDs.txt"
alt_data = np.load("/u/cs401/A1/feats/Alt_feats.dat.npy")
left_data = np.load("/u/cs401/A1/feats/Left_feats.dat.npy")
right_data = np.load("/u/cs401/A1/feats/Right_feats.dat.npy")
center_data = np.load("/u/cs401/A1/feats/Center_feats.dat.npy")

bristol_dic = {}
ratings_warriner_dic = {}
alt_list = []
left_list = []
right_list = []
center_list = []
with open(alt) as alt_f:
    for line in alt_f:
        alt_list.append(line.strip())
with open(left) as left_f:
    for line in left_f:
        left_list.append(line.strip())
with open(right) as right_f:
    for line in right_f:
        right_list.append(line.strip())
with open(center) as center_f:
    for line in center_f:
        center_list.append(line.strip())
with open(bristol) as BG_file:
    csv_reader = csv.reader(BG_file)
    for row in csv_reader:
        # get rid of header
        if row[0] != "Source":
            k = row[1]
            bristol_dic[k] = row
with open(ratings_warriner) as RW_file:
    csv_reader = csv.reader(RW_file)
    for row in csv_reader:
        # get rid of header
        if row[1] != "Word":
            k = row[1]
            ratings_warriner_dic[k] = row

#from handout
first_person_list = ["i", "me", "my", "mine", "we", "us", "our", "ours"]
second_person_list = ["you", "your", "yours", "u", "ur", "urs"]
third_person_list = ["he", "him", "his", "she", "her", "hers", "it", "its", "they", "them", "their", "theirs"]
cc = "CC"
past_tens_list = ["VBD", "VBN"]
slang_list = ["smh", "fwb", "lmfao", "lmao", "lms", "tbh", "rofl", "wtf", "bff", "wyd", "lylc", "brb","atm", "imao", "sml", "btw", "imho", "fyi", "ppl", "sob", "ttyl", "imo", "ltr", "thx",
"kk", "omg", "ttys", "afn", "bbs", "cya", "ez", "f2f", "gtr", "ic", "jk", "k", "ly","ya", "nm", "np", "plz", "ru", "so", "tc", "tmi", "ym", "ur", "u", "sol", "lol", "fml"]
future_tense_list = ["’ll", "will", "gonna"]
common_noun_list = ["NN", "NNS"]
proper_noun_list = ["NNP", "NNPS"]
wh_word_list = ["WDT", "WP", "WP$", "WRB"]
adverb_list = ["RB", "RBR", "RBS"]
puc_str = string.punctuation
pattern = re.compile("[" + puc_str + "]{2,}")

def extract1( comment ):
    ''' This function extracts features from a single comment

    Parameters:
        comment : string, the body of a comment (after preprocessing)

    Returns:
        feats : numpy Array, a 173-length vector of floating point features (only the first 29 are expected to be filled, here)
    '''
    feat = np.zeros(174)
    # 1. Number of first-person pronouns
    # 2. Number of second-person pronouns
    # 3. Number of third-person pronouns
    # 4. Number of coordinating conjunctions
    # 5. Number of past-tense verbs
    # 6. Number of future-tense verbs
    # 7. Number of commas
    # 8. Number of multi-character punctuation tokens
    # 9. Number of common nouns
    # 10. Number of proper nouns
    # 11. Number of adverbs
    # 12. Number of wh- words
    # 13. Number of slang acronyms
    # 14. Number of words in uppercase (≥ 3 letters long)
    first_person_count = 0
    second_person_count = 0
    third_person_count = 0
    coordinating_conjunctions_count = 0
    past_tens_count = 0
    future_tense_count = 0
    commas_count = 0
    multi_character_count = 0
    common_nouns_count = 0
    proper_nouns_count = 0
    adverbs_count = 0
    wh_word_count = 0
    slang_count = 0
    three_letters_long_count = 0
    average_sentence_count = 0
    splitedComm = comment.split()
    for word_with_tag in splitedComm:
        split_word = word_with_tag.split("/")
        word = split_word[0]
        tag = split_word[1]
        if word in first_person_list:
            first_person_count += 1
        elif word in second_person_list:
            second_person_count += 1
        elif word in third_person_list:
            third_person_count += 1
        elif tag == cc:
            coordinating_conjunctions_count += 1
        elif tag in past_tens_list:
            past_tens_count += 1
        elif word in future_tense_list:
            future_tense_count += 1
        elif tag == ",":
            commas_count += 1
        elif len(pattern.findall(word)) > 0:
            multi_character_count += 1
        elif tag in common_noun_list:
            common_nouns_count +=1
        elif tag in proper_noun_list:
            proper_nouns_count += 1
        elif tag in adverb_list:
            adverbs_count += 1
        elif tag in wh_word_list:
            wh_word_count += 1
        elif word in slang_list:
            slang_count += 1
        elif len(word) > 3 and word.isupper():
            three_letters_long_count+=1
    # fill with all features
    feat[0] = first_person_count
    feat[1] = second_person_count
    feat[2] = third_person_count
    feat[3] = coordinating_conjunctions_count
    feat[4] = past_tens_count
    feat[5] = future_tense_count
    feat[6] = commas_count
    feat[7] = multi_character_count
    feat[8] = common_nouns_count
    feat[9] = proper_nouns_count
    feat[10] = adverbs_count
    feat[11] = wh_word_count
    feat[12] = slang_count
    feat[13] = three_letters_long_count
    
    # 15. Average length of sentences, in tokens
    split_sentence = comment.split("\n")
    num_sentences = len(split_sentence)
    num_words = 0
    for sentence in split_sentence:
        words = sentence.split(" ")
        num_words += len(words)
    if num_sentences == 0:
        feat[14] = 0
    else:
        feat[14] = num_words / num_sentences
   
   # 16. Average length of tokens, excluding punctuation-only tokens, in characters
    num_without_punc = 0
    num_total_tokens = 0
    for sentence in split_sentence:
        tokens = sentence.split(" ")
        num_total_tokens += len(tokens)
        for token in tokens:
            if token.split("/")[0].isalnum():
                num_without_punc+=len(token.split("/")[0])
    if num_total_tokens == 0:
        feat[15] = 0
    else:
        feat[15] = num_without_punc / num_total_tokens
    # 17. Number of sentences. done
    feat[16] = num_sentences 
    # 18. Average of AoA (100-700) from Bristol, Gilhooly, and Logie norms
    num_total_AoA = 0
    list_AoA = []
    num_aoa_words = 0
    for word_with_tag in splitedComm:
        split_word = word_with_tag.split("/")
        word = split_word[0]
        if word in bristol_dic:
            num_aoa_words+=1
            list_AoA.append(int(bristol_dic[word][3]))
            num_total_AoA += int(bristol_dic[word][3])
    if num_aoa_words == 0:
        feat[17] = 0
    else:
        feat[17] = num_total_AoA / num_aoa_words
    # 19. Average of IMG from Bristol, Gilhooly, and Logie norms
    num_total_IMG = 0
    list_IMG = []
    num_IMG_words = 0
    for word_with_tag in splitedComm:
        split_word = word_with_tag.split("/")
        word = split_word[0]
        if word in bristol_dic:
            num_IMG_words+=1
            list_IMG.append(int(bristol_dic[word][4]))
            num_total_IMG += int(bristol_dic[word][4])
    if num_IMG_words == 0:
        feat[18] = 0
    else:
        feat[18] = num_total_IMG / num_IMG_words
    # 20. Average of FAM from Bristol, Gilhooly, and Logie norms
    num_total_FAM = 0
    list_FAM = []
    num_FAM_words = 0
    for word_with_tag in splitedComm:
        split_word = word_with_tag.split("/")
        word = split_word[0]
        if word in bristol_dic:
            num_FAM_words+=1
            list_FAM.append(int(bristol_dic[word][5]))
            num_total_FAM += int(bristol_dic[word][5])
    if num_FAM_words == 0:
        feat[19] = 0
    else:
        feat[19] = num_total_FAM / num_FAM_words
    # 21. Standard deviation of AoA (100-700) from Bristol, Gilhooly, and Logie norms
    if list_AoA == []:
        feat[20] = 0
    else:
        feat[20] = np.std(list_AoA)
    # 22. Standard deviation of IMG from Bristol, Gilhooly, and Logie norms
    if list_IMG == []:
        feat[21] = 0
    else:
        feat[21] = np.std(list_IMG)
    # 23. Standard deviation of FAM from Bristol, Gilhooly, and Logie norms
    if list_FAM== []:
        feat[22] = 0
    else:    
        feat[22] = np.std(list_FAM)
    # 24. Average of V.Mean.Sum from Warringer norms
    num_total_V_mean_sum = 0.0
    list_V_mean = []
    num_V_mean_words = 0
    for word_with_tag in splitedComm:
        split_word = word_with_tag.split("/")
        word = split_word[0]
        if word in ratings_warriner_dic:
            num_V_mean_words +=1
            list_V_mean.append(float(ratings_warriner_dic[word][2]))
            num_total_V_mean_sum += float(ratings_warriner_dic[word][2])
    if num_V_mean_words == 0:
        feat[23] = 0
    else:
        feat[23] = num_total_V_mean_sum / num_V_mean_words
    # 25. Average of A.Mean.Sum from Warringer norms
    num_total_A_mean_sum = 0.0
    list_A_mean = []
    num_A_mean_words = 0
    for word_with_tag in splitedComm:
        split_word = word_with_tag.split("/")
        word = split_word[0]
        if word in ratings_warriner_dic:
            num_A_mean_words +=1
            list_A_mean.append(float(ratings_warriner_dic[word][5]))
            num_total_A_mean_sum += float(ratings_warriner_dic[word][5])
    if num_A_mean_words == 0:
        feat[24] = 0
    else:
        feat[24] = num_total_A_mean_sum / num_A_mean_words
    # 26. Average of D.Mean.Sum from Warringer norms
    num_total_D_mean_sum = 0.0
    list_D_mean = []
    num_D_mean_words = 0
    for word_with_tag in splitedComm:
        split_word = word_with_tag.split("/")
        word = split_word[0]
        if word in ratings_warriner_dic:
            num_D_mean_words +=1
            list_D_mean.append(float(ratings_warriner_dic[word][8]))
            num_total_D_mean_sum += float(ratings_warriner_dic[word][8])
    if num_D_mean_words == 0:
        feat[25] = 0
    else:
        feat[25] = num_total_D_mean_sum / num_D_mean_words
    # 27. Standard deviation of V.Mean.Sum from Warringer norms
    if list_V_mean == []:
        feat[26] = 0
    else:
        feat[26] = np.std(list_V_mean)
    # 28. Standard deviation of A.Mean.Sum from Warringer norms
    if list_A_mean == []:
        feat[27] = 0
    else:
        feat[27] = np.std(list_A_mean)
    # 29. Standard deviation of D.Mean.Sum from Warringer norms
    if list_D_mean == []:
        feat[28] = 0
    else:
        feat[28] = np.std(list_D_mean)



    # print(feat)


    return feat
    # TODO: your code here

def main( args ):
    data = json.load(open(args.input))
    feats = np.zeros( (len(data), 173+1))

    i = 0
    while i < len(data):
        feats[i] = extract1(data[i]["body"])
        if data[i]["cat"] == "Left":
            feats[i][173] = 0
            liwc_index = left_list.index(data[i]["id"])
            feats[i][29:173] = left_data[liwc_index]
        elif data[i]["cat"] == "Center":
            feats[i][173] = 1
            liwc_index = center_list.index(data[i]["id"])
            feats[i][29:173] = center_data[liwc_index]
        elif data[i]["cat"] == "Right":
            feats[i][173] = 2
            liwc_index = right_list.index(data[i]["id"])
            feats[i][29:173] = right_data[liwc_index]
        else:
            feats[i][173] = 3
            liwc_index = alt_list.index(data[i]["id"])
            feats[i][29:173] = alt_data[liwc_index]
        i+=1
    np.savez_compressed( args.output, feats)

    
if __name__ == "__main__": 

    parser = argparse.ArgumentParser(description='Process each .')
    parser.add_argument("-o", "--output", help="Directs the output to a filename of your choice", required=True)
    parser.add_argument("-i", "--input", help="The input JSON file, preprocessed as in Task 1", required=True)
    args = parser.parse_args()
                 

    main(args)

