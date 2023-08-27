import numpy as np
import pandas as pd
import nltk
import seaborn as sn
import matplotlib.pyplot as plt
import language_tool_python
import scipy.stats as stats

from sklearn import metrics, model_selection
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import xgboost as xgb

from utils import compute_metrics, convert_to_decimal, extract_answer, extract_editDistance_fixed, extract_mean, extract_nbKeystroke, extract_std, plot_confusion_matrix


tool = language_tool_python.LanguageTool('en-US')


def is_bad_rule(rule): return rule.message == 'Possible spelling mistake found.' and len(
    rule.replacements) and rule.replacements[0][0].isupper()


# data import
df_fixed = pd.read_csv(
    'Dataset/emosurv/Fixed_Text_Typing_Dataset.csv', sep=';')
df_freq = pd.read_csv('Dataset/emosurv/Frequency_Dataset.csv', sep=';')
df_user = pd.read_csv('Dataset/emosurv/Participants_Information.csv', sep=';')

df_freq = df_freq.rename(columns={'User ID': 'userId'})

# reconstruct users' sentences from the keycode in df_fixed
sentence = ''
uppercase = False
uppercase_tmp = False
i_start = 0
i_end = 0
text_fixed = pd.DataFrame(columns=['sentence', 'idx_start', 'idx_end'])

df_fixed.keyCode = df_fixed.keyCode.astype(str)

for i in range(len(df_fixed)):
    # df_fixed._id[i][:-2] != df_fixed._id[i-1][:-2]:
    if i > 0 and (df_fixed.userId[i] != df_fixed.userId[i-1] or df_fixed.emotionIndex[i] != df_fixed.emotionIndex[i-1]):
        i_end = i-1
        if len(text_fixed) == 62:
            ids_start = [i for i in range(
                len(sentence)) if sentence[i:i+4] == 'once']
            text_fixed = text_fixed._append(pd.DataFrame(
                [[sentence, i_start, i_start+ids_start[1]-1]], columns=['sentence', 'idx_start', 'idx_end']), ignore_index=True)
            text_fixed = text_fixed._append(pd.DataFrame(
                [[sentence, i_start+ids_start[1], i_start+ids_start[2]-1]], columns=['sentence', 'idx_start', 'idx_end']), ignore_index=True)
            text_fixed = text_fixed._append(pd.DataFrame(
                [[sentence, i_start+ids_start[2], i_end]], columns=['sentence', 'idx_start', 'idx_end']), ignore_index=True)
        elif len(text_fixed) == 134 or len(text_fixed) == 190:
            ids_start = [i for i in range(
                len(sentence)) if sentence[i:i+4] == 'Once']
            text_fixed = text_fixed._append(pd.DataFrame(
                [[sentence, i_start, i_start+ids_start[1]-1]], columns=['sentence', 'idx_start', 'idx_end']), ignore_index=True)
            text_fixed = text_fixed._append(pd.DataFrame(
                [[sentence, i_start+ids_start[1], i_end]], columns=['sentence', 'idx_start', 'idx_end']), ignore_index=True)
        else:
            if sentence[0].islower() and sentence[1].isupper():
                sentence = sentence.swapcase()
            text_fixed = text_fixed._append(pd.DataFrame([[sentence, i_start, i_end]], columns=[
                                            'sentence', 'idx_start', 'idx_end']), ignore_index=True)
        i_start = i
        sentence = ''
        uppercase = False
        uppercase_tmp = False
    if df_fixed.keyCode[i] == '\\b':    # delete key
        sentence = sentence[:max(0, len(sentence)-1)]
        uppercase_tmp = False
        continue
    if df_fixed.keyCode[i] == '\\u0014':    # caps lock key
        uppercase = not uppercase
        continue
    if df_fixed.keyCode[i] == '\\u0010' and not uppercase:     # shift key
        uppercase_tmp = True
        continue
    if df_fixed.keyCode[i] == '¼':     # comma key
        sentence += ','
        uppercase_tmp = False
        continue
    if df_fixed.keyCode[i] == 'ß':     # exclamation mark key
        sentence += '!'
        uppercase_tmp = False
        continue
    if df_fixed.keyCode[i] == '¾':     # dot key
        sentence += '.'
        uppercase_tmp = False
        continue
    if df_fixed.keyCode[i] == '4    ':     # apostrophe key
        sentence += "'"
        uppercase_tmp = False
        continue
    if df_fixed.keyCode[i] == '6    ':     # dash key
        sentence += "-"
        uppercase_tmp = False
        continue
    if df_fixed.keyCode[i] == '¿':     # colon key
        sentence += ":"
        uppercase_tmp = False
        continue
    if uppercase:
        sentence += df_fixed.keyCode[i].upper()
        continue
    if uppercase_tmp:
        sentence += df_fixed.keyCode[i].upper()
        uppercase_tmp = False
        continue
    sentence += df_fixed.keyCode[i]

text_fixed = text_fixed._append(pd.DataFrame([[sentence, i_start, i]], columns=[
                                'sentence', 'idx_start', 'idx_end']), ignore_index=True)

# fixed text that users have been asked to write

gold_fixed = {
    'N': 'Once there was a cat and a mouse. Usually, cats eat mice, and mice run away from cats. But this cat and this mouse liked each other very much. They liked each other so much that they lived together.',
    'H': 'We can not help falling in love with cute and funny babies. Their beautiful and joyful laughter makes us happy.',
    'C': 'Beautiful nature and calm music are always relaxing. The soft sounds are so pleasant to listen to. Watching the superb nature calms our body and soul.',
    'S': 'The boy yells at his father to wake him up. But the father is dead. The poor boy is very sad. He realized that his father is gone forever.',
    'A': 'Jake has a horrible temper, especially when he drinks alcohol. He gets andgry and agressive when he is drunk. He savagely  beats and violeates his wife.'
}

# functions to extract features from the df_fixed table
df_fixed = df_fixed.drop(['D1U3', 'D1D3'], axis=1)

# print(df_fixed.head())

key_features = ['D1U1', 'D1U2', 'D1D2', 'U1D2', 'U1U2']

for column in key_features:
    df_fixed[column] = df_fixed[column].apply(convert_to_decimal)

text_fixed['userId'] = df_fixed.userId[text_fixed.idx_start].values
text_fixed['emotionIndex'] = df_fixed.emotionIndex[text_fixed.idx_start].values
text_fixed = text_fixed.reindex(
    columns=['idx_start', 'idx_end', 'userId', 'emotionIndex', 'sentence'])
text_fixed['editDistance'] = text_fixed.apply(lambda x: extract_editDistance_fixed(
    gold_fixed, x['emotionIndex'], x['sentence']), axis=1)
text_fixed['nbKeystroke'] = text_fixed.apply(lambda x: extract_nbKeystroke(
    df_fixed.index, x['idx_start'], x['idx_end']), axis=1)
text_fixed['answer'] = text_fixed.apply(lambda x: extract_answer(
    df_fixed.answer, x['idx_start'], x['idx_end']), axis=1)

for feat in key_features:
    df_fixed[feat] = df_fixed[feat].apply(
        lambda x: np.nan if np.abs(x) > 1570000000000 else x)
    text_fixed[feat+'_mean'] = text_fixed.apply(lambda x: extract_mean(
        df_fixed[feat], x['idx_start'], x['idx_end']), axis=1)
    text_fixed[feat+'_std'] = text_fixed.apply(lambda x: extract_std(
        df_fixed[feat], x['idx_start'], x['idx_end']), axis=1)

# print(text_fixed.head())

# filter fixed-text experiments in df_freq
df_freq_fixed = df_freq[df_freq.textIndex == 'FI'].reset_index(drop=True)

# correction of expections to align df_freq_fixed and text_fixed tables
text_fixed.loc[44, 'userId'] = 94
tmp = text_fixed.loc[30]
text_fixed.loc[30] = text_fixed.loc[31]
text_fixed.loc[31] = tmp

# alignment of df_freq_fixed and text_fixed tables
text_fixed['text_index'] = -1
df_freq_fixed['text_index'] = -1
index = 0

# print(text_fixed.info())

i = 0
j = 0
while i < len(df_freq_fixed) and j < len(text_fixed):
    if df_freq_fixed.userId[i] == text_fixed.userId[j] and df_freq_fixed.emotionIndex[i] == text_fixed.emotionIndex[j]:
        df_freq_fixed.loc[i, 'text_index'] = index
        text_fixed.loc[j, 'text_index'] = index
        index += 1
        i += 1
        j += 1
    elif j != len(text_fixed)-1 and df_freq_fixed.userId[i] == text_fixed.userId[j+1] and df_freq_fixed.emotionIndex[i] == text_fixed.emotionIndex[j+1]:
        df_freq_fixed.loc[i, 'text_index'] = index
        text_fixed.loc[j+1, 'text_index'] = index
        index += 1
        i += 1
        j += 2
    elif i != len(df_freq_fixed)-1 and df_freq_fixed.userId[i+1] == text_fixed.userId[j] and df_freq_fixed.emotionIndex[i+1] == text_fixed.emotionIndex[j]:
        df_freq_fixed.loc[i+1, 'text_index'] = index
        text_fixed.loc[j, 'text_index'] = index
        index += 1
        i += 2
        j += 1
    else:
        i += 1
        j += 1

# print(df_freq_fixed.info())

# correction of expectations to align df_freq_fixed and df_user tables
tmp = df_user.loc[92]
df_user.loc[92] = df_user.loc[91]
df_user.loc[91] = tmp

# alignment of df_freq_index and df_user tables
df_freq_fixed['user_index'] = -1
index = 0

j = 0
for i in range(len(df_freq_fixed)):
    if i in [39, 59, 71, 145, 156, 159, 179, 235]:
        j += 1
    if j < len(df_user) and df_freq_fixed.userId[i] == df_user.userId[j]:
        df_freq_fixed.loc[i, 'user_index'] = j
        i += 1
    else:
        if j < len(df_user)-1 and df_freq_fixed.userId[i] == df_user.userId[j+1]:
            df_freq_fixed.loc[i, 'user_index'] = j+1
            i += 1
            j += 1
        else:
            for k in range(1, 6):
                if df_freq_fixed.userId[i] == df_freq_fixed.userId[i-k]:
                    df_freq_fixed.loc[i,
                                      'user_index'] = df_freq_fixed.loc[i-k, 'user_index']
                    break

# print(df_user.info())

# merge all tables
df_fixed_all = df_freq_fixed.join(
    text_fixed, on='text_index', how='left', rsuffix='_right')

df_fixed_all = df_fixed_all.drop(
    ['userId_right', 'emotionIndex_right', 'text_index_right'], axis=1)
df_fixed_all = df_fixed_all.join(df_user.reset_index().rename(
    columns={'index': 'user_index'}), on='user_index', how='left', rsuffix='_right')
df_fixed_all = df_fixed_all.drop(['user_index_right', 'userId_right'], axis=1)

# drop more non useful columns
df_fixed_all = df_fixed_all.drop(['userId', 'textIndex', 'text_index', 'user_index', 'idx_start',
                                  'idx_end', 'answer', 'pcTimeAverage', 'typistType', 'status', 'degree',
                                  'country'], axis=1)

df_fixed_all.to_csv('Dataset/emosurv/Fixed_Text_Typing_Dataset_mod.csv')
print(df_fixed_all.info())
