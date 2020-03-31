import numpy as np
from scipy import sparse

class Bigram:
    def __init__(self, language):
        self.language = language
    
    def fit(self, v, good_turing_threshold = 20):
        num_words = max(v.c[self.language].vocabulary_.values()) + 1
        self.counts = [None, 
                       np.zeros(num_words), 
                       sparse.dok_matrix((num_words, num_words), dtype = int)]
        for key_vector in v.key_vectors[self.language]:
            for bigram in self.to_ngrams(key_vector, 2):
                self.counts[2][bigram] += 1
            for unigram in key_vector:
                self.counts[1][unigram] += 1
        max_count = max(self.counts[2].values())
        count_of_counts = np.zeros(max_count + 1)
        for n in self.counts[2].values():
            count_of_counts[n] += 1 
        count_of_counts[0] = num_words ** 2 - self.counts[2].sum()
        self.ratio = count_of_counts[1:good_turing_threshold + 1] / count_of_counts[:good_turing_threshold]

    def to_ngrams(self, key_vector, n):
        k = tuple(key_vector)
        return [k[i:i + n] for i in range(len(k) - n + 1)]
    
    def probability(self, key_vector):
        bigrams = self.to_ngrams(key_vector, 2)
        product = 1
        for bigram in bigrams:
            bigram_count = self.counts[2][bigram]
            unigram_count = self.counts[1][bigram[0]]
            if bigram_count < len(self.ratio):
                bigram_count = (bigram_count + 1) * self.ratio[bigram_count]
            product = product * bigram_count / unigram_count
        return product
    
    