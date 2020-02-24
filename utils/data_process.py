from utils.tokenizing import preprocess_br
import os
import pandas as pd
# 处理summary或者description没有任何内容的情况
def dealNan(x):
    if type(x) == float or type(x) == list:
        x = ' '
    return x

def clean_pandas(data):
    data.rename(columns=lambda x: x.strip(), inplace=True)
    data['summary'] = data.summary.apply(dealNan)
    data['description'] = data.description.apply(dealNan)
    # 对文本数据进行清洗
    data['summary'] = data['summary'].map(lambda x: preprocess_br(x))
    data['description'] = data['description'].map(lambda x: preprocess_br(x))

    return data

def read_data_from_csv(project):
    datasets = {
        'ambari': 'ambari.csv',
        'camel': 'camel.csv',
        'chromium': 'chromium.csv',
        'derby': 'derby.csv',
        'wicket': 'wicket.csv',
        'mozilla_large': 'mozilla_large.csv',
        'chromium_large': 'chromium_large.csv'
    }
    if project not in datasets:
        raise ValueError

    # 保存数据清洗结果
    pandas_data_file = os.path.join('..', 'resources', 'dataset_pd', project)
    if os.path.exists(pandas_data_file):
        df_all = pd.read_pickle(pandas_data_file)
    else:
        data_file = os.path.join('..', 'resources',project, datasets[project])
        df_all = pd.read_csv(data_file, sep=',', encoding='ISO-8859-1')
        # clean the textual fileds
        df_all = clean_pandas(df_all)
        df_all.to_pickle(pandas_data_file)
    return df_all
