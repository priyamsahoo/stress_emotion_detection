import numpy as np
import pandas as pd
from utils import convert_to_decimal, extract_answer, extract_mean, extract_nbKeystroke, extract_std
import nltk
import seaborn as sn
import matplotlib.pyplot as plt
import language_tool_python
import scipy.stats as stats

tool = language_tool_python.LanguageTool('en-US')
is_bad_rule = lambda rule: rule.message == 'Possible spelling mistake found.' and len(rule.replacements) and rule.replacements[0][0].isupper()

# functions to extract features from the df_free table
def extract_editDistance_free(sentence):
    matches = tool.check(sentence)
    matches = [rule for rule in matches if not is_bad_rule(rule)]
    correct = language_tool_python.utils.correct(sentence, matches)
    return nltk.edit_distance(sentence, correct)

# data import
# data import
df_free = pd.read_csv('Dataset/emosurv/Free_Text_Typing_Dataset.csv', sep=';')
df_freq = pd.read_csv('Dataset/emosurv/Frequency_Dataset.csv', sep=';')
df_user = pd.read_csv('Dataset/emosurv/Participants_Information.csv', sep=';')

df_free = df_free.rename(columns={'userid':'userId'})
df_freq = df_freq.rename(columns={'User ID':'userId'})

# sentence reconstruction
# reconstruct users' sentences from the keycode in df_free

sentence = ''
uppercase = False
uppercase_tmp = False
i_start = 0
i_end = 0
text_free = pd.DataFrame(columns=['sentence', 'idx_start', 'idx_end'])

df_free.keyCode = df_free.keyCode.astype(str)

for i in range(len(df_free)):
    if i > 0 and (df_free.userId[i] != df_free.userId[i-1] or df_free.emotionIndex[i] != df_free.emotionIndex[i-1]): #df_free._id[i][:-2] != df_free._id[i-1][:-2]:
        i_end = i-1
        if len(sentence) > 1 and sentence[0].islower() and sentence[1].isupper():
            sentence = sentence.swapcase()
        text_free = text_free._append(pd.DataFrame([[sentence, i_start, i_end]], columns=['sentence', 'idx_start', 'idx_end']), ignore_index = True)
        i_start = i
        sentence = ''
        uppercase = False
        uppercase_tmp = False
    if df_free.keyCode[i] == '\\b':   # delete key
        sentence = sentence[:max(0,len(sentence)-1)]
        uppercase_tmp = False
        continue
    if df_free.keyCode[i] == '\\u0014':   # caps lock key
        uppercase = not uppercase
        continue
    if df_free.keyCode[i] == '\\u0010' and not uppercase:   # shift key
        uppercase_tmp = True
        continue
    if df_free.keyCode[i] == '¼':     # comma key
        sentence += ','
        uppercase_tmp = False
        continue
    if df_free.keyCode[i] == 'ß':     # exclamation mark key
        sentence += '!'
        uppercase_tmp = False
        continue
    if df_free.keyCode[i] == '¾':   # dot key
        sentence += '.'
        uppercase_tmp = False
        continue
    if df_free.keyCode[i] == '4    ':   # apostrophe key
        sentence += "'"
        uppercase_tmp = False
        continue
    if df_free.keyCode[i] == '6    ':    # dash key
        sentence += "-"
        uppercase_tmp = False
        continue
    if df_free.keyCode[i] == '¿':   # colon key
        sentence += ":"
        uppercase_tmp = False
        continue
    if uppercase:
        sentence += df_free.keyCode[i].upper()
        continue
    if uppercase_tmp:
        sentence += df_free.keyCode[i].upper()
        uppercase_tmp = False
        continue
    sentence += df_free.keyCode[i]    

text_free = text_free._append(pd.DataFrame([[sentence, i_start, i]], columns=['sentence', 'idx_start', 'idx_end']), ignore_index = True)

# drop features highly correlated  with other features
df_free = df_free.drop(['D1U3','D1D3'], axis=1)

key_features = ['D1U1','D1U2','D1D2','U1D2','U1U2']

for column in key_features:
    df_free[column] = df_free[column].apply(convert_to_decimal)

text_free['userId'] = df_free.userId[text_free.idx_start].values
text_free['emotionIndex'] = df_free.emotionIndex[text_free.idx_start].values
text_free = text_free.reindex(columns=['idx_start','idx_end','userId','emotionIndex','sentence'])
text_free['editDistance'] = text_free.apply(lambda x: extract_editDistance_free(x['sentence']), axis=1)
text_free['nbKeystroke'] = text_free.apply(lambda x: extract_nbKeystroke(df_free.index, x['idx_start'], x['idx_end']), axis=1)
text_free['answer'] = text_free.apply(lambda x: extract_answer(df_free.answer, x['idx_start'], x['idx_end']), axis=1)

for feat in key_features:
    df_free[feat] = df_free[feat].apply(lambda x: np.nan if np.abs(x) > 1570000000000 else x)
    text_free[feat+'_mean'] = text_free.apply(lambda x: extract_mean(df_free[feat], x['idx_start'], x['idx_end']), axis=1)
    text_free[feat+'_std'] = text_free.apply(lambda x: extract_std(df_free[feat], x['idx_start'], x['idx_end']), axis=1)


# print(text_free.head())

df_for_sentiment = text_free.drop(text_free.columns.difference(['userId', 'emotionIndex', 'sentence']), axis=1)

print(df_for_sentiment.head())

df_for_sentiment.to_csv('emosurv_free_text_for_sentiment.csv')