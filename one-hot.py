# 单词级one-hot编码

import numpy as np
import string
from keras.preprocessing.text import Tokenizer
samples = ['The cat sat on the mat.', 'The dog ate my homework.']

token_index_word = {}
for sample in samples:
    for word in sample.split():
        if word not in token_index_word:
            token_index_word[word] = len(token_index_word) + 1

max_length = 10

results = np.zeros(shape=(len(samples), max_length, max(token_index_word.values()) + 1))

for i, sample in enumerate(samples):
    for j, word in list(enumerate(sample.split()))[:max_length]:
        index = token_index_word.get(word)
        results[i, j, index] = 1

print(token_index_word)
print(results)

print("===============================")

# 字符集one-hot编码


characters = string.printable
token_index = dict(zip(range(1, len(characters) + 1), characters))

max_length = 50
results = np.zeros((len(samples), max_length, max(token_index.keys()) + 1))
for i, sample in enumerate(samples):
        for j, character in enumerate(sample):
                index = token_index.get(character)
                results[i, j, index] = 1

print(token_index)
print(results)

print("===============================")

# Keras单词级one-hot编码

tokenizer = Tokenizer(num_words=1000)# 只考虑前1000个单词，避免处理非常大的输入向量空间
tokenizer.fit_on_texts(samples)

# print(tokenizer.word_index)

sequences = tokenizer.texts_to_sequences(samples)# 将文本之间转化成对应的索引数字序列
print(sequences)

one_hot_results = tokenizer.texts_to_matrix(samples, mode='binary')

print(one_hot_results)

word_index = tokenizer.word_index

print(word_index)