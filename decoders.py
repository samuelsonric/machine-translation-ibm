import numpy as np
from itertools import permutations

## Greedy Decoder
def decode(f, tm, lm, v, n):
    f = np.array(v.to_keys(f, 'f'))
    e = np.array(v.to_keys('null', 'e') + [tm.top_n_translations(word, 'f', 1)[0] for word in f])
    a = np.array([i for i in range(1, len(f) + 1)])
    p = noisy_probability(e, f, a, tm, lm)
    for i in range(n):
        print(v.from_keys(e[1:], 'e'))
        e_steps = []
        a_steps = []
        p_steps = []
        algorithms = [translate_one_or_two_words, translate_and_insert, remove_word_of_fertility_zero, swap_segments, join_words]

        for algorithm in algorithms:
            e_step, a_step, p_step = algorithm(e, f, a, tm, lm)
            e_steps.append(e_step)
            a_steps.append(a_step)
            p_steps.append(p_step)
        if max(p_steps) <= p:
            break
        best_step = np.argmax(p_steps)
        e = e_steps[best_step]
        a = a_steps[best_step]
        p = p_steps[best_step]
        
def translate_one_or_two_words(e, f, a, tm, lm):
    n = 10
    max_p = 0
    max_e = None
    max_a = None
    for j_1 in range(len(f)):
        for word_1 in tm.top_n_translations(f[j_1], 'f', n):
            e_1, a_1 = change_translation(word_1, j_1, e, f, a, tm, lm)
            for j_2 in range(len(f)):
                for word_2 in tm.top_n_translations(f[j_2], 'f', n):
                    e_2, a_2 = change_translation(word_2, j_2, e_1, f, a_1, tm, lm)
                    p = noisy_probability(e_2, f, a_2, tm, lm)
                    if p > max_p:
                        max_p = p
                        max_e = e_2
                        max_a = a_2
    return (max_e, max_a, max_p)

def translate_and_insert(e, f, a, tm, lm):
    n = 10
    m = 10
    max_p = 0
    max_e = None
    max_a = None
    for j in range(len(f)):
        for word_1 in tm.top_n_translations(f[j], 'f', n):
            e_1, a_1 = change_translation(word_1, j, e, f, a, tm, lm)
            for word_2 in tm.low_fertility_words:
                e_2, a_2 = best_insert(word_2, None, e_1, f, a_1, tm, lm)
                p = noisy_probability(e_2, f, a_2, tm, lm)
                if p > max_p:
                    max_p = p
                    max_e = e_2
                    max_a = a_2
    return (max_e, max_a, max_p)

def remove_word_of_fertility_zero(e, f, a, tm, lm):
    fertility_zero = set(range(1, len(e))) - set(a)
    max_p = 0
    max_e = None
    max_a = None
    for i in fertility_zero:
        e_i, a_i = remove_word(i, 0, e, a)
        p = noisy_probability(e_i, f, a_i, tm, lm)
        if p > max_p:
            max_p = p
            max_e = e_i
            max_a = a_i  
    return (max_e, max_a, max_p)

def swap_segments(e, f, a, tm, lm):
    n = 7
    max_p = 0
    max_e = None
    max_a = None
    if len(e) == 1:
        return (e, f, p)
    elif len(e) <= n:
        for permutation in permutations(range(1, len(e))):
            permutation = list((0,) + permutation)
            e_permuted = e[permutation]
            a_permuted = np.array([permutation[i] for i in a])
            p = noisy_probability(e_permuted, f, a_permuted, tm, lm)
            if p > max_p:
                max_p = p
                max_e = e_permuted
                max_a = a_permuted
    else:
        for i_1 in range(1, len(e) - 1):
            for i_2 in range(i_1, len(e - 1)):
                for k_1 in range(i_2 + 1, len(e)):
                    for k_2 in range(k_1, len(e)):
                        permutation = np.array(range(len(e)))
                        permutation = permutation[np.r_[0:i_1, k_1:k_2 + 1, i_2 + 1:k_1, i_1:i_2 + 1, k_2 + 1:len(e)]]
                        e_permuted = e[permutation]
                        a_permuted = np.array([permutation[i] for i in a])
                        p = noisy_probability(e_permuted, f, a_permuted, tm, lm)
                        if p > max_p:
                            max_p = p
                            max_e = e_permuted
                            max_a = a_permuted
    return (max_e, max_a, max_p)   

def join_words(e, f, a, tm, lm):
    max_p = 0
    max_e = None
    max_a = None
    for i_1 in range(len(e)):
        for i_2 in range(1, len(e)):
            if not i_1 == i_2:
                e_joined, a_joined = remove_word(i_2, i_1, e, a)
                p = noisy_probability(e_joined, f, a_joined, tm, lm)
                if p > max_p:
                    max_p = p
                    max_e = e_joined
                    max_a = a_joined
    return (max_e, max_a, max_p)

##realign french word at j to word
def change_translation(word, j, e, f, a, tm, lm):
    if not word == e[a[j]]:
        if word == e[0]:
            if (a == a[j]).sum() == 1:
                e, a = remove_word(a[j], 0, e, a)
            else:
                a = a.copy()
                a[j] = 0
        elif a[j] == 0:
            e, a = best_insert(word, j, e, f, a, tm, lm)
        else:
            e = e.copy()
            e[a[j]] = word
    return (e, a)

##remove word at position i from e
##all french words aligned with the removed word are realigned to word at r
def remove_word(i, r, e, a):
    e = e[np.r_[:i, i + 1:len(e)]]
    a = a + (a == i) * (r - i)
    a = a - (a > i)
    return (e, a)

##optimal insertion of word into e, with j realigned to word
##j = None for no realignment
def best_insert(word, j, e, f, a, tm, lm):
    max_p = 0
    for i in range(len(e)):
        e_i, a_i = insert_word(word, i, e, a)
        if not j == None:
            a_i[j] = i
        p = noisy_probability(e_i, f, a_i, tm, lm)
        if p >= max_p:
            max_p = p
            max_e = e_i
            max_a = a_i
    return (max_e, max_a)

##insert word into e after position i
def insert_word(word, i, e, a):
    e = np.r_[e[:i + 1], word, e[i + 1:]]
    a = a + (a > i)
    return (e, a)

def noisy_probability(e, f, a, tm, lm):
    return tm.probability(e, f, a) * lm.probability(e)