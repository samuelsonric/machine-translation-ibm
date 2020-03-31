from sklearn.feature_extraction.text import CountVectorizer

class Vectorizer:
    def __init__(self, sample, from_language, to_language):
        self.from_language = from_language
        self.to_language = to_language
        self.sample = sample
        self.c = {}
        self.count_vectors = {}
        self.analyzer = {}
        self.key_vectors = {}
        self.oh_vectors = {}
        self.backwards_vocabulary = {}
    
    def fit_count_vectorizer(self):
        for language in self.sample:
            self.c[language] = CountVectorizer(token_pattern = r"(?u)\b[\w\']+\b")
            self.c[language].fit(self.sample[language])
            
    def fit_count_vectors(self):
        if not self.c:
            self.fit_count_vectorizer()
        for language in self.sample:
            self.count_vectors[language] = self.c[language].transform(self.sample[language]).T
            
    def build_analyzer(self):
        if not self.c:
            self.fit_count_vectorizer()
        for language in self.sample:
            self.analyzer[language] = self.c[language].build_analyzer()
            
    def to_tokens(self, string, language):
        if not self.analyzer:
            self.build_analyzer()
        return self.analyzer[language](string)
        
    def to_keys(self, string, language):
        tokens = self.to_tokens(string, language)
        return [self.c[language].vocabulary_[word] for word in tokens]
    
    def build_backwards_vocabulary(self, language):
        self.backwards_vocabulary[language] = {self.c[language].vocabulary_[word] : word for word in self.c[language].vocabulary_}
    
    def from_keys(self, key_vector, language):
        if not language in self.backwards_vocabulary:
            self.build_backwards_vocabulary(language)
        return ' '.join([self.backwards_vocabulary[language][key] for key in key_vector])
    
    def from_key(self, key, language):
        return self.from_keys([key], language)
                
    def fit_key_vectors(self):
        for language in self.sample:
            self.key_vectors[language] = []
            for string in self.sample[language]:
                self.key_vectors[language].append(self.to_keys(string, language))
        
    def fit_one_hot_vectors(self):
        for language in self.sample:
            self.oh_vectors[language] = []
            for string in self.sample[language]:
                tokens = self.to_tokens(string, language)
                self.oh_vectors[language].append(self.c[language].transform(tokens))   
                
    def filter_vectors(self, min_length = 2, max_length = 50, max_ratio = 3):
        if not self.key_vectors:
            self.fit_key_vectors()
        to_remove = []
        for i in range(len(self.sample[self.from_language])):
            lengths = [len(self.key_vectors[self.from_language][i]) - 1, len(self.key_vectors[self.to_language][i])]
            less_than_n = [not min_length <= l <= max_length for l in lengths]
            if sum(less_than_n) or max(lengths) > max_ratio * min(lengths):
                to_remove.append(i)
        to_remove = to_remove[::-1]
        for i in to_remove:
            for language in self.sample:
                del self.sample[language][i]
        self.fit_key_vectors()
        if self.count_vectors:
            self.fit_count_vectors()
        if self.oh_vectors:
            self.fit_one_hot_vectors()
        