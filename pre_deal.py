from io import open
import unicodedata
import re
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义语言类,将指定语言中的词汇映射成数值

SOS_token = 0
EOS_token = 1
class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.index2word[self.n_words] = word
            self.n_words += 1

#字符规范化(unicode转Ascii)
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

#加载数据到内存
data_path="data/eng-fra.txt"
def readLangs(lang1, lang2, reverse=False):
    lines = open(data_path, 'r',encoding='utf-8').\
        read().strip().split('\n')
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]
    input_lang = Lang(lang1)
    output_lang = Lang(lang2)
    return input_lang, output_lang, pairs


lang1 = 'eng'
lang2 = 'fra'
input_lang, output_lang, pairs = readLangs(lang1, lang2)
'''
测试模块
print('input_lang', input_lang)
print('output_lang', output_lang)
print('pairs前五个', pairs[:5])
'''

#过滤符合要求的语言对
#现在取小样本的翻译对，设置句子单词或标点的最多个数
MAX_LENGTH = 10
#选带有指定前缀的语言特征为训练数据
eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s ",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)


































