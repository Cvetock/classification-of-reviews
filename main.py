import sys
import numpy as np
from collections import Counter
import math

f = open('reviews.txt')
raw_reviews = f.readlines()
f.close()

f = open('labels.txt')
raw_labels = f.readlines()
f.close()

tokens = list(map(lambda x:set(x.split(" ")),raw_reviews))

vocab = set()
for sent in tokens:
    for word in sent:
        if(len(word)>0):
            vocab.add(word)
vocab = list(vocab)

word2index = {}
for i,word in enumerate(vocab):
    word2index[word]=i

input_dataset = list()
for sent in tokens:
    sent_indices = list()
    for word in sent:
        try:
            sent_indices.append(word2index[word])
        except:
            ""
    input_dataset.append(list(set(sent_indices)))

target_dataset = list()
for label in raw_labels:
    if label == 'positive\n':
        target_dataset.append(1)
    else:
        target_dataset.append(0)

np.random.seed(1)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


alpha, iterations = (0.01, 2)
hidden_size = 100

weights_0_1 = 0.2 * np.random.random((len(vocab), hidden_size)) - 0.1
weights_1_2 = 0.2 * np.random.random((hidden_size, 1)) - 0.1

correct, total = (0, 0)

for iter in range(iterations):
    for i in range(len(input_dataset) - 1000):  # Обучение на первых 24 000 обзорах
        x, y = (input_dataset[i], target_dataset[i])  # Векторное представление + sigmoid
        layer_1 = sigmoid(np.sum(weights_0_1[x], axis=0))
        layer_2 = sigmoid(np.dot(layer_1, weights_1_2))  # Линейный слой + softmax

        layer_2_delta = layer_2 - y  # Разность между прогнозом и истиной
        layer_1_delta = layer_2_delta.dot(weights_1_2.T)  # Обратное распространение

        weights_0_1[x] -= layer_1_delta * alpha
        weights_1_2 -= np.outer(layer_1, layer_2_delta) * alpha

        if np.abs(layer_2_delta) < 0.5:
            correct += 1
            total += 1

            if (i % 10 == 9):
                progress = str(i / float(len(input_dataset)))
                sys.stdout.write('\rIter:' + str(iter) + \
                                 ' Progress:' + progress[2:4] + \
                                 '.' + progress[4:6] + \
                                 ' % Training Accuracy:' + \
                                 str(correct / float(total)) + '%')

            print()

    correct, total = (0, 0)

    for i in range(len(input_dataset) - 1000, len(input_dataset)):
        x = input_dataset[i]
        y = target_dataset[i]
        layer_1 = sigmoid(np.sum(weights_0_1[x], axis=0))
        layer_2 = sigmoid(np.dot(layer_1, weights_1_2))

        if (np.abs(layer_2 - y) < 0.5):
            correct += 1
            total += 1

    print("Test Accuracy:" + str(correct / float(total)))

def similar(target='beautiful'):
    target_index = word2index[target]
    scores = Counter()
    for word,index in word2index.items():
        raw_difference = weights_0_1[index] - (weights_0_1[target_index])
        squared_difference = raw_difference * raw_difference
        scores[word] = -math.sqrt(sum(squared_difference))

    return scores.most_common(10)
print(similar('beautiful'))