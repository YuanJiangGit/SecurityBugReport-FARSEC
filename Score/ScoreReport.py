import numpy as np
class ScoreReport:
    def __init__(self):
        pass

    def score_report(self,scored_words,report):
        '''
        compute the score of report
        :param scored_words:
        :param report:
        :return:
        '''
        result=[scored_words[word] for word in report if word in scored_words]
        result_=[1-score for score in result]

        # multiply continuously
        top=np.prod(result)
        bottom=top+np.prod(result_)
        return top/bottom

