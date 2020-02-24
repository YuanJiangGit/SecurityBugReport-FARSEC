import json
import math
from collections import defaultdict
from utils.data_process import read_data_from_csv
import operator
import pandas as pd
import os

class SecurityWords:
    def __init__(self,project):
        self.project=project

    def read_data(self,project):
        self.project = project
        with open('../resources/'+self.project+'/'+self.project+'.json') as f:
            sbr_corpus = json.load(f)
        return sbr_corpus

    def load_sbr_corpus(self,df_train):
        # the form 1/2 as training data

        self.SBR = df_train[df_train['Security'] == 1]
        # obtain the sbr corpus
        SBRs = self.SBR['summary'] + self.SBR['description']
        sbr_corpus = []
        for index in range(len(SBRs)):
            term_freq = defaultdict(int)
            SBR = SBRs.iloc[index]
            for term in SBR:
                term_freq[term] += 1
            sbr_corpus.append(term_freq)
        return sbr_corpus

    def idf(self,corpus,term):
        '''
        计算term的idf值
        :param corpus:
        :param term:
        :return:
        '''
        N=len(corpus)
        contain_term_sbrs=[SBR for SBR in corpus if term in SBR]
        # 包含terms的sbr的个数
        N_t = len(contain_term_sbrs)
        idf_t = math.log(N / (N_t + 1))
        return idf_t

    def get_idf_cache(self,corpus):
        '''
        计算corpus中所有term的idf值
        :param corpus:
        :return:
        '''
        terms_set=set()
        Tag = [terms_set.add(term) for SBR in corpus for term in SBR]
        idf_cache={term:self.idf(corpus,term) for term in terms_set}
        return idf_cache

    def tf(self,term_freq,max_freq):
        '''
        计算term的tf值
        :param term_freq: term在某个SBR中出现的频次
        :param max_freq:  某个SBR中出现频次最大的值
        :return:
        '''
        return (0.5*term_freq)/max_freq+0.5


    def tf_idf(self,idf_value,term_freq,max_freq):
        return self.tf(term_freq,max_freq)*idf_value


    def tf_idf_pair(self,idf_cache,pair):
        '''
        计算每个安全缺陷报告的tf_idf值
        :param idf_cache:
        :param pair:
        :return:
        '''
        # the maximum frequence in pair(SBR)
        max_freq=max(pair.values())
        terms_tf_idf={}
        for term,freq in pair.items():
            idf_value = idf_cache[term]
            tf_idf_value = self.tf_idf(idf_value, freq, max_freq)
            terms_tf_idf[term]=tf_idf_value
        return terms_tf_idf

    def top_security_words(self,corpus):
        idf_cache=self.get_idf_cache(corpus)
        corpus_tf_idf=defaultdict(float)
        for pair in corpus:
            tf_idf_ = self.tf_idf_pair(idf_cache, pair)
            for term,tf_idf in tf_idf_.items():
                corpus_tf_idf[term] += tf_idf
        # sort
        corpus_tf_idf=sorted(corpus_tf_idf.items(), key=operator.itemgetter(1),reverse=True)
        return corpus_tf_idf[:100]

if __name__ == '__main__':
    # ambari 92 derby 95 camel 89 wicket 86 chromium 93
    # ['derby','camel','wicket','chromium']  ['chromium_large','mozilla_large']
    # projects=['ambari','derby','camel','wicket','chromium']
    projects=['ambari','derby','camel','wicket','chromium']
    for project in projects:
        # 加载全部数据集
        df_all = read_data_from_csv(project)
        df_train = pd.DataFrame(df_all, index=range(int(len(df_all) / 2)))
        sw=SecurityWords(project)
        corpus = sw.load_sbr_corpus(df_train)
        sbr_words = sw.top_security_words(corpus)
        sbr_words = [tuple[0] for tuple in sbr_words]
        print(project)

        with open('../resources/SBR_words/' + project, 'r') as f:
            true_sbr_words = f.read().split()
        intersec_words = [word for word in true_sbr_words if word in sbr_words]
        # print(intersec_words)
        print(len(intersec_words))

