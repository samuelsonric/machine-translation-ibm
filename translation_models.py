from sklearn.preprocessing import normalize
from scipy import sparse
import numpy as np
import matplotlib.pyplot as plt
import pickle

class IBM1:
    def __init__(self, from_language, to_language):
        self.from_language = from_language
        self.to_language = to_language
        
    def build_model(self, v, threshold = 0.05):
        if not v.count_vectors:
            v.fit_count_vectors()
        record_of_errors = []
        self.translation = normalize(v.count_vectors[self.from_language] @ v.count_vectors[self.to_language].T, 'l1')
        error = 1
        while error > threshold:
            self.translation, error = self.EM(v.count_vectors[self.from_language], v.count_vectors[self.to_language], self.translation)
            record_of_errors.append(error)
        self.length = self.build_length(v.count_vectors[self.from_language].sum(0).flat, v.count_vectors[self.to_language].sum(0).flat)
        self.build_low_fertility_words(v)
        plt.plot(record_of_errors, color = 'r')
        plt.axhline(threshold, color = 'k')
        plt.show()
    
    def EM(self, E, F, translation):
        R = E.T @ translation
        R.data = 1 / R.data
        t = translation.multiply(E @ (F.T.multiply(R)))
        t = normalize(t, 'l1')
        error = np.max(abs(translation - t))
        return (t, error)
    
    def build_length(self, E, F):
        L = max(E) + 1
        M = max(F) + 1
        length = np.zeros([L, M])
        for i in range(len(E)):
            length[E[i], F[i]] += 1
        return normalize(length, 'l1')
    
    def build_low_fertility_words(self, v, n = 100):
        if not v.count_vectors:
            v.fit_count_vectors()
        if not v.key_vectors:
            v.fit_key_vectors()
        mean_fertility = np.zeros(self.translation.shape[0])
        for e, f in zip(v.key_vectors[self.from_language], v.key_vectors[self.to_language]):
            a = self.viterbi(e, f)
            for i in range(len(e)):
                mean_fertility[e[i]] += (a == i).sum()
        mean_fertility = (1 + mean_fertility) / (1 + np.array(v.count_vectors['e'].sum(1).T).reshape(-1))**2
        self.low_fertility_words = set(mean_fertility.argsort()[:n]) - set(v.to_keys('null', 'e'))
        
    def save_low_fertility_words(self, path):
        if not path.endswith('.pickle'):
            path = path + '.pickle'
        with open(path, 'wb') as handle:
            pickle.dump(self.low_fertility_words, handle)
    
    def load_low_fertility_words(self, path):
        with open(path, 'rb') as handle:
            self.low_fertility_words = pickle.load(path, handle)                    
                             
    def save_translation_matrix(self, path):
        sparse.save_npz(path, self.translation)
        
    def load_translation_matrix(self, path):
        self.translation = sparse.load_npz(path)
        
    def save_length_matrix(self, path):
        np.save(path, self.length)
        
    def load_length_matrix(self, path):
        self.length = np.load(path)
    
    def probability(self, from_key_vector, to_key_vector, a):
        l = len(from_key_vector)
        m = len(to_key_vector)
        return self.length[l, m] / (l ** m) * np.prod([self.translation[from_key_vector[a[j]], word] for j, word in enumerate(to_key_vector)])
    
    def top_n_translations(self, key, language, n = 1):
        if language == self.from_language:
            return self.translation[key].toarray().reshape(-1).argsort()[:-n - 1:-1]
        else:
            return self.translation[:, key].toarray().reshape(-1).argsort()[:-n - 1:-1]
        
    def viterbi(self, from_key_vector, to_key_vector):
        return np.array(self.translation[from_key_vector][:, to_key_vector].argmax(0)).reshape(-1)
    
class IBM2:
    def __init__(self, from_language, to_language):
        self.from_language = from_language
        self.to_language = to_language
        
    def build_model(self, v, tm, threshold = 0.05):
        if not v.oh_vectors:
            v.fit_one_hot_vectors()
        record_of_errors = {'translation' : [], 'alignment' : []}
        self.length = tm.length
        self.translation = tm.translation
        L = max(len(e) for e in v.key_vectors[self.from_language])
        M = max(len(f) for f in v.key_vectors[self.to_language])
        self.alignment = np.array([[normalize(sparse.csr_matrix(np.ones((m + 1, l + 1))), 'l1') for l in range(L)] for m in range(M)])
        translation_error = alignment_error = 1
        while translation_error > threshold or alignment_error > 0.5:
            self.translation, self.alignment, translation_error, alignment_error = self.EM(v.key_vectors[self.from_language],
                                                                                           v.key_vectors[self.to_language],
                                                                                           v.oh_vectors[self.from_language],
                                                                                           v.oh_vectors[self.to_language],
                                                                                           self.translation,
                                                                                           self.alignment)
            record_of_errors['translation'].append(translation_error)
            record_of_errors['alignment'].append(alignment_error)
        self.build_low_fertility_words(v)
        plt.plot(record_of_errors['translation'], color = 'r')
        plt.plot(record_of_errors['alignment'], color = 'b')
        plt.axhline(threshold, color = 'k')
        plt.show()
    
    def EM(self, E, F, EINV, FINV, translation, alignment):
        t_list = []
        a = np.empty(alignment.shape, dtype = object)
        for e, f, einv, finv in zip(E, F, EINV, FINV):
            m, l = len(f), len(e)
            if l == 0 or m == 0:
                continue
            N = normalize(alignment[m - 1, l - 1].multiply(translation[e][:, f].T), 'l1')
            if a[m - 1, l - 1] == None:
                a[m - 1, l - 1] = N
            else:
                a[m - 1, l - 1] += N
            t_list.append((finv.T @ N @ einv).T)
        t = sparse.csr_matrix(self.sum_sparse(t_list))
        t = normalize(t, 'l1')
        translation_error = np.max(abs(translation - t))
        alignment_error = np.zeros_like(alignment)
        for m in range(a.shape[0]):
            for l in range(a.shape[1]):
                if a[m, l] != None and alignment[m, l] != None:
                    a[m, l] = normalize(a[m, l], 'l1')
                    alignment_error[m, l] = np.max(abs(alignment[m, l] - a[m, l]))               
        alignment_error = np.max(alignment_error)
        return (t, a, translation_error, alignment_error)
    
    def sum_sparse(self, m):
        x = np.zeros(m[0].shape,m[0].dtype)
        for a in m:
            x[a.nonzero()] += a.data
        return x
    
    def build_low_fertility_words(self, v, n = 100):
        if not v.count_vectors:
            v.fit_count_vectors()
        if not v.key_vectors:
            v.fit_key_vectors()
        mean_fertility = np.zeros(self.translation.shape[0])
        for e, f in zip(v.key_vectors[self.from_language], v.key_vectors[self.to_language]):
            a = self.viterbi(e, f)
            for i in range(len(e)):
                mean_fertility[e[i]] += (a == i).sum()
        mean_fertility = (1 + mean_fertility) / (1 + np.array(v.count_vectors['e'].sum(1).T).reshape(-1))**2
        self.low_fertility_words = set(mean_fertility.argsort()[:n]) - set(v.to_keys('null', 'e'))
                             
    def save_low_fertility_words(self, path):
        if not path.endswith('.pickle'):
            path = path + '.pickle'
        with open(path, 'wb') as handle:
            pickle.dump(self.low_fertility_words, handle)
    
    def load_low_fertility_words(self, path):
        with open(path, 'rb') as handle:
            self.low_fertility_words = pickle.load(path, handle) 
    
    def save_translation_matrix(self, path):
        sparse.save_npz(path, self.translation)
        
    def load_translation_matrix(self, path):
        self.translation = sparse.load_npz(path)
        
    def save_alignment_matrix(self, path):
        sparse.save_npz(path, self.alignment)
        
    def load_alignment_matrix(self, path):
        self.alignment = np.load(path, allow_pickle = True)
        
    def load_length_matrix(self, path):
        self.length = np.load(path)
        
    def probability(self, from_key_vector, to_key_vector, a):
        l = len(from_key_vector)
        m = len(to_key_vector)
        return self.length[l, m] * np.prod([self.translation[from_key_vector[a[j]], word] * self.alignment[m - 1, l - 1][j, a[j]] for j, word in enumerate(to_key_vector)])
    
    def top_n_translations(self, key, language, n = 1):
        if language == self.from_language:
            return self.translation[key].toarray().reshape(-1).argsort()[:-n - 1:-1]
        else:
            return self.translation[:, key].toarray().reshape(-1).argsort()[:-n - 1:-1]
        
    def viterbi(self, from_key_vector, to_key_vector):
        l = len(from_key_vector)
        m = len(to_key_vector)
        return np.array(self.translation[from_key_vector][:, to_key_vector].multiply(self.alignment[m - 1, l - 1].T).argmax(0)).reshape(-1)
