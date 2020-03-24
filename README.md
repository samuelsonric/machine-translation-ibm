# Statistical Machine Translation Using IBM Translation Models

An ongoing project to translate between English and French.

## Introduction

The IBM translation models are a sequence of statistical machine translation algorithms that date to the late 1980s. The models were created as part of IBM's "Candide" project, which aimed to automatically translate French and English sentences. As an excercise in theory, I decided to try implementing the models myself. The goal of this implementation is efficiency: I want to build a learning algorithm that can be practically trained on home computers.

The IBM models work by approximating the conditional probability that a given English sentence is the translation of a given French sentence. The models learn the conditional distribution by reading pairs of translated English and French sentences. Once the distribution is learned, a translater translates a French sentence to whichever English sentence is most likely. 

Learning a conditional translation distribution over pairs of French and English sentences is difficult because of the sparsity of the data. One is unlikely to see the same French sentence translated into English two different ways, and unless one vectorizes input sentences very cleverly, there is little relationship between the the Euclidian distance of two sentence-vectors and the semantic distance of their linguistic counterparts. Also, unless one vectorizes very cleverly, any conditional probability matrix between French and English sentences would be much too large to fit in a computer.

Vectorizing very cleverly may well solve these problems. Systems like Word2Vec exist for vectorizing text with a meaningful inner product. This was not the approach of IBM. Rather, the researchers made strict assumptions about the range of possible conditional distributions, and then performed Maximum Likelihood Estimation to find the likeliest distribution in that range.

This strategy produces translators that do not adequately discriminate against ill-formed English sentences. To compensate for this, IBM translated using a "noisy channel model". Rather than computing directly the conditional probability $P(e|f)$ that a French sentence $f$ translates to an english sentence $e$, the researchers preferred to compute the Bayesian equivalent $\frac{P(f|e)P(e)}{P(f)}$. Given a French sentence, the translator chooses the English sentence that maximizes the numerator of the above fraction. This approach puts the emphasis on well-formedness on the distribution $P(e)$ over english sentences. Though this method is more computationally intensive (one must learn two distributions instead of one), it tends to produce better results.

## IBM Model 1

