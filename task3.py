import re
import numpy as np
from matplotlib import mlab
import time
from pymystem3 import Mystem
from sklearn import model_selection, svm
from matplotlib import pyplot as plt

def vowel(token):
    q_vowels = 0
    for let in token:
        if let in 'уеыаоэёяию':
            q_vowels += 1
    return q_vowels


def unique(words):
    all_letters = ''
    for w in words:
        w = w.strip('<>!@"\';:/%^&*()-_#{}[],.?\\—').lstrip('<>!@"\';:/%^&*()-_#{}[],.?\\—')
        for l in w: 
            if w not in all_letters:
                all_letters += w
    return len(set(all_letters))

def features_collect(cla, corpus):
    result = []
    for line in corpus:
        line = line.strip().lower()
        line = line.split(' ')
        if len(line) > 0:
            uni = unique(line)
            letters = [len(word.strip('<>!@"\';:/%^&*()-_#{}[],.?\\—').lstrip('<>!@"\';:/%^&*()-_#{}[],.?\\—')) for word in line if len(word) > 0 and word[0] not in 'qwertyuiopasdfghjklzxcvbnm']
            vowels = [vowel(word1) for word1 in line if len(word1) > 0]
            if len(letters) > 0: 
                result.append([cla, np.sum(letters), uni, np.sum(vowels), np.median(letters), np.median(vowels)])
    return result


def main():
    with open('anna.txt', encoding='utf-8') as f:
        anna = f.read()
    with open('sonets.txt', encoding='utf-8') as f:
        sonets = f.read()
    anna_sentences = re.split(r'(?:[.]\s*){3}|[.?!]', anna)
    sonet_sentences = re.split(r'(?:[.]\s*){3}|[.?!]', sonets)
    anna_data = features_collect(1, anna_sentences)
    sonet_data = features_collect(2, sonet_sentences)
    anna_data = np.array(anna_data)
    sonet_data = np.array(sonet_data)
    data = np.vstack((anna_data, sonet_data))
    p = mlab.PCA(data[:, 1:], True)
    N = len(anna_data)
    plt.figure()
    plt.plot(p.Y[:N,0], p.Y[:N,1], 'og', p.Y[N:,0], p.Y[N:,1], 'sb')
    plt.show()
    clf = svm.LinearSVC(C=0.1)
    clf.fit(data[::2, 1:], data[::2, 0])
    print('Done')
    print(clf.score(data[::2, 1:], data[::2, 0]))
    wrong = 0
    for obj in data[1::2, :]:
        label = clf.predict(obj[1:].reshape(1, -1))
        if label != obj[0] and wrong < 3:
            print('Пример ошибки машины: class = ', obj[0], ', label = ', label, ', экземпляр ', obj[1:])
            wrong += 1
        if wrong > 3:
            break


if __name__ == '__main__':
    main()


