import fasttext as ft
import numpy as np 
import sys
from scipy.spatial.distance import cosine
import nltk
from nltk.corpus import stopwords

#word_list = open("xxx.y.txt", "r")
stops = set(stopwords.words('english'))


# from https://stackoverflow.com/questions/21030391/how-to-normalize-array-numpy
def normalized(a, axis=-1, order=2):
    """Utility function to normalize the rows of a numpy array."""
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2==0] = 1
    return a / np.expand_dims(l2, axis)


# from Smith et al., 2017
def make_training_matrices(source_dictionary, target_dictionary, bilingual_dictionary):
    """
    Source and target dictionaries are the FastVector objects of
    source/target languages. bilingual_dictionary is a list of 
    translation pair tuples [(source_word, target_word), ...].
    """
    source_matrix = []
    target_matrix = []

    for (source, target) in bilingual_dictionary:
        if source in source_dictionary and target in target_dictionary:
            source_matrix.append(source_dictionary[source])
            target_matrix.append(target_dictionary[target])

    # return training matrices
    return np.array(source_matrix), np.array(target_matrix)

def learn_transformation(source_matrix, target_matrix, normalize_vectors=True):
    """
    Source and target matrices are numpy arrays, shape
    (dictionary_length, embedding_dimension). These contain paired
    word vectors from the bilingual dictionary.
    """
    # optionally normalize the training vectors
    if normalize_vectors:
        source_matrix = normalized(source_matrix)
        target_matrix = normalized(target_matrix)

    # perform the SVD
    product = np.matmul(source_matrix.transpose(), target_matrix)
    U, s, V = np.linalg.svd(product)

    # return orthogonal transformation which aligns source language to the target
    return np.matmul(U, V)

def get_nearest_words(vector, model, words): 
    distances = []
    for w in words : 
        #dist = np.linalg.norm(vector - model[w])
        dist = cosine(vector, model[w])
        distances.append([w, dist])
    
    distances.sort(key=lambda x:x[1])
    return distances[:10]


def get_words(filename): 
    intersection = []
    with open(filename) as f : 
        for line in f : 
            intersection.append(line.rstrip("\n"))

    return intersection 


if __name__ == "__main__" : 

    """
    Example Usage : 
    python find_alignment.py <list of top words> <source model> <dest model> <save filepath> <number of words to check the alignment on (sorted by decreasing frequency)> 

    python find_alignment.py 2020_top_10k_words.txt models/cnn_2020_97M.bin models/fox_2020_97M.bin 2020_fox_cnn_97.txt 5000

    """

    intersection = get_words(sys.argv[1])
    source_model_filepath = sys.argv[2]
    target_model_filepath = sys.argv[3]
    result_save_path = sys.argv[4]
    num_translate = int(sys.argv[5])

    bilingual_dict = [(w,w) for w in stops]
    source_model = ft.load_model(source_model_filepath)
    target_model = ft.load_model(target_model_filepath)
    print("Models loaded")

    source_matrix, target_matrix = make_training_matrices(
        source_model , target_model, bilingual_dict
    )

    # learn and apply the transformations : 
    transform = learn_transformation(source_matrix, target_matrix)


    count = 0 
    index = 0 
  

    with open(result_save_path, "w") as fp: 
        print(f"index\tcount\talign%\tsource\ttarget\trest")   
        for word in intersection: 
            index += 1
            if index <= num_translate : 
                
                #print(word)
                source_word =  source_model[word]
                #fox = fox_model[word]
                
                
                n = get_nearest_words(np.dot(source_word, transform), target_model, intersection)
                n1 = get_nearest_words(source_word, source_model, intersection)

                if n[0][0] != word : 
                    count += 1
                    
                #print(word)
                
                s = f"{index}\t{count}\t{(index-count)/index:.3f}\t{word}\t{n[0][0]}\t"
                s2 = "\t".join([x[0] for x in n1])
                s1 = "\t".join([x[0] for x in n])

                fp.write(s+s2+"\t|\t"+s1+"\n")
                print(s+s1)
                    
                #s = f"{index}\t{count}\t{(index-count)/index:.3f}\t{word} mapped to {n[0][0]}\n"   
                
            else : 
                break 