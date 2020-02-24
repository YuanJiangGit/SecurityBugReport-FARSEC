import pandas as pd
import math
class ScoreKeywords:
    def __init__(self):
        pass

    def term_freq(self,corpus,term):
        contain_term_brs = [SBR for SBR in corpus if term in SBR]
        n=len(contain_term_brs)+1
        return n

    def get_freqs_cache(self,corpus):
        '''
        corpus中每个term在文档中出现的频次
        :param corpus:
        :return: {term: freq(term)}
        '''
        terms_set = set()
        Tag = [terms_set.add(term) for SBR in corpus for term in SBR]
        freqs_cache={term:self.term_freq(corpus,term) for term in terms_set}
        return freqs_cache

    def score_word_two(self,good,bad,word):
        '''
        compute the score of one term (word) in feature set
        :param good: nsbr
        :param bad: sbr
        :param word: feature set中的一个安全相关的关键词
        :return:  score
        '''
        # good is dict, if word in dict.keys(), then return good[word], else return 0
        g = good.get(word,0)
        b = bad.get(word,0)*2
        nbad=len(bad)
        ngood=len(good)
        score=min(1,b*1.0/nbad)/(min(1,b*1.0/nbad)+min(1,g*1.0/ngood))
        score = min(0.99, max(0.01, score))
        return score

    def score_word_sq(self,good,bad,word):
        g = good.get(word,0)
        b = math.pow(bad.get(word,0),2)
        nbad = len(bad)
        ngood = len(good)
        score = min(1, b * 1.0 / nbad) / (min(1, b * 1.0 / nbad) + min(1, g * 1.0 / ngood))
        score = min(0.99, max(0.01, score))
        return score

    def score_word_none(self,good,bad,word):
        g = good.get(word,0)
        b = bad.get(word,0)
        nbad = len(bad)
        ngood = len(good)
        score = min(1, b * 1.0 / nbad) / (min(1, b * 1.0 / nbad) + min(1, g * 1.0 / ngood))
        score = min(0.99, max(0.01, score))
        return score

    def score_word(self,identify):
        '''
        "Score filters in three ways, which affect the scores of the reports and therefore the number of filtered reports.
           identify indicates which one is used.
           1) farsec (none)   = treats SBRs and NSBRs equally.
           2) farsectwo (two) = adds bias by multiplying the number of SBRs by 2.
           3) farsecsq (sq)   = adds bias by squaring the number of SBRs.
        :param identify:
        :return:
        '''
        if identify not in ['two','sq','none']:
            return None
        if identify=='two':
            return self.score_word_two
        elif identify=='sq':
            return self.score_word_sq
        else:
            return self.score_word_none


    def score_words(self,df_train,key_farsec,top_words):
        sbr_corpus = df_train[df_train['Security'] == 1]
        nsbr_corpus = df_train[df_train['Security'] == 0]

        sbr_cache=self.get_freqs_cache(sbr_corpus['summary']+sbr_corpus['description'])
        nsbr_cache = self.get_freqs_cache(nsbr_corpus['summary'] + nsbr_corpus['description'])

        # 根据key_farsec,得到计算keyword 分数的函数
        score_word_fn=self.score_word(key_farsec)
        scored_words = {term: score_word_fn(nsbr_cache, sbr_cache, term) for term in top_words}
        return scored_words
