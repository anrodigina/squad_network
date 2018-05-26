import argparse
import collections
import multiprocessing

import spacy

global nlp
nlp = spacy.load('en', parser=False)

import tensorflow as tf
import numpy as np
import msgpack

import re
import unicodedata

embedding = np.load('./data/embedding.npy')

import msgpack
with open('./data/meta.msgpack', 'rb') as f:
  vocab = msgpack.load(f, encoding='utf8')

w2id = {w: i for i, w in enumerate(vocab['vocab'])}
tag2id = {w: i for i, w in enumerate(vocab['vocab_tag'])}
ent2id = {w: i for i, w in enumerate(vocab['vocab_ent'])}

def clean_spaces(text):
    """normalize spaces in a string."""
    text = re.sub(r'\s', ' ', text)
    return text


def normalize_text(text):
    return unicodedata.normalize('NFD', text)

def annotate(row, wv_cased):
    global nlp
    id_, context, question = row[:3]
    q_doc = nlp(clean_spaces(question))
    c_doc = nlp(clean_spaces(context))
    question_tokens = [normalize_text(w.text) for w in q_doc]
    context_tokens = [normalize_text(w.text) for w in c_doc]
    question_tokens_lower = [w.lower() for w in question_tokens]
    context_tokens_lower = [w.lower() for w in context_tokens]
    context_token_span = [(w.idx, w.idx + len(w.text)) for w in c_doc]
    context_tags = [w.tag_ for w in c_doc]
    context_ents = [w.ent_type_ for w in c_doc]
    question_lemma = {w.lemma_ if w.lemma_ != '-PRON-' else w.text.lower() for w in q_doc}
    question_tokens_set = set(question_tokens)
    question_tokens_lower_set = set(question_tokens_lower)
    match_origin = [w in question_tokens_set for w in context_tokens]
    match_lower = [w in question_tokens_lower_set for w in context_tokens_lower]
    match_lemma = [(w.lemma_ if w.lemma_ != '-PRON-' else w.text.lower()) in question_lemma for w in c_doc]
    # term frequency in document
    counter_ = collections.Counter(context_tokens_lower)
    total = len(context_tokens_lower)
    context_tf = [counter_[w] / total for w in context_tokens_lower]
    context_features = list(zip(match_origin, match_lower, match_lemma, context_tf))
    if not wv_cased:
        context_tokens = context_tokens_lower
        question_tokens = question_tokens_lower
    return (id_, context_tokens, context_features, context_tags, context_ents,
            question_tokens, context, context_token_span) + row[3:]
  
def to_id(row, w2id, tag2id, ent2id, unk_id=1):
    context_tokens = row[1]
    context_features = row[2]
    context_tags = row[3]
    context_ents = row[4]
    question_tokens = row[5]
    question_ids = [w2id[w] if w in w2id else unk_id for w in question_tokens]
    context_ids = [w2id[w] if w in w2id else unk_id for w in context_tokens]
    tag_ids = [tag2id[w] for w in context_tags]
    ent_ids = [ent2id[w] for w in context_ents]
    return (row[0], context_ids, context_features, tag_ids, ent_ids, question_ids) + row[6:]


import keras
import numpy as np
from keras.models import Model
from keras.layers import Input, Dense, Flatten, LSTM, Bidirectional, Concatenate, Add, Reshape, Multiply, Dot
from keras.layers import SimpleRNN, GRU, BatchNormalization, Activation, RepeatVector,Permute, TimeDistributed
from keras.layers import multiply, average, BatchNormalization, Average
from keras.layers.wrappers import TimeDistributed, Bidirectional
from keras import backend as K
from keras.layers import Lambda, Masking

from keras.activations import softmax

txt_len = 767
qst_len = 60
vector_dim = 300


hidden = 128

input_qst = Input(shape=(qst_len, vector_dim), name='in_qst')
input_txt = Input(shape=(txt_len, vector_dim), name='in_txt')
input_features = Input(shape=(txt_len, 4), name='in_features')
input_tags = Input(shape=(txt_len, 50), name='in_tags')
input_ent = Input(shape=(txt_len, 19), name='in_ents')

in_txt = Concatenate() ([input_txt, input_features, input_tags, input_ent])

txt_1 = Bidirectional(LSTM(hidden,
                return_sequences=True
#               ,  return_state=True
                ), name='txt1') (in_txt)

txt_1_norm = BatchNormalization() (txt_1)

txt_2 = Bidirectional(LSTM(hidden,
                return_sequences=True,
#                 return_state=True
                ), name='txt2') (txt_1_norm)

txt_2_norm = BatchNormalization() (txt_2)


qst_1 =  Bidirectional(LSTM(hidden,
                return_sequences=True
                ), name='qst1') (input_qst)

qst_1_norm = BatchNormalization() (qst_1)

qst_2 = Bidirectional(LSTM(hidden,
                return_sequences=True,
                ), name='qst2') (qst_1_norm)
qst_2_norm = BatchNormalization() (qst_2)

txt_result = Concatenate(name='concat_txt') ([txt_1_norm, txt_2_norm]);


attention = Dense(1, activation='tanh')(qst_2_norm)
attention1 = Reshape((qst_len,))(attention)
attention2 = Activation('softmax')(attention1)
attention3 = RepeatVector(2 * hidden)(attention2) 
attention4 = Permute([2, 1])(attention3)

out_attent = Multiply(name='mul') ([qst_2_norm, attention]) #mmm


tmp_sum = Lambda(lambda x: K.sum(x, axis=-2), name='tmp_sum') (out_attent)


txt_a = Dense(hidden * 2, use_bias=False, activation='softmax', name='txt_a') (txt_result)
txt_b = Dense(hidden * 2, use_bias=False, activation='softmax', name='txt_b') (txt_result)

bb = Lambda(lambda x1: K.batch_dot(x1[0], x1[1])) ([txt_a, tmp_sum])
ee = Lambda(lambda x1: K.batch_dot(x1[0], x1[1])) ([txt_b, tmp_sum])

begin = Activation('softmax', name='ans_beg') (bb)
end = Activation('softmax', name='ans_end') (ee)

model = Model(inputs=[input_txt, input_qst, input_features, input_tags, input_ent], outputs=[begin, end])

model.compile(
    loss='categorical_crossentropy',
    optimizer='RMSprop',
    metrics=['accuracy']
)

model.load_weights('./models/last-v-4.h5')

while True:
    try:
        while True:
            context = input('Enter your context, please\n_________________________________________________\n')
            print('_________________________________________________\n')
            if context.strip():
                break
        while True:
            question = input('Enter your question, please\n_________________________________________________\n')
            print('_________________________________________________\n')
            if question.strip():
                break
    except EOFError:
        print('##############################')
        break
    annotated = annotate(('interact-{}'.format(1), context, question), vocab['wv_cased'])
    model_in = to_id(annotated, w2id, tag2id, ent2id)
    context_data = np.zeros((2, 767), dtype=int)
    qst_data = np.zeros((2,60), dtype=int)

    for i in range(len(model_in[1])):
      context_data[0][i] = model_in[1][i]

    for i in range(len(model_in[5])):
      qst_data[0][i] = model_in[5][i]

    cont_features = np.zeros((2, 767, 4), dtype=float)

    for i in range(len(model_in[2])):
      cont_features[0][i][0] = float(model_in[2][i][0])
      cont_features[0][i][1] = float(model_in[2][i][1])
      cont_features[0][i][2] = float(model_in[2][i][2])
      cont_features[0][i][3] = model_in[2][i][3]  

    con_tag = np.zeros((2, 767), dtype=int)
    for i in range(len(model_in[1])):
      con_tag[0][i] = model_in[3][i]

    ent_tag = np.zeros((2, 767), dtype=int)
    for i in range(len(model_in[1])):
      ent_tag[0][i] = model_in[4][i]

    voc_ent = np.zeros((19, 19), dtype=float)
    for i in range(19):
      voc_ent[i][i] = 1.
    voc_con = np.zeros((50, 50), dtype=float)
    for i in range(50):
      voc_con[i][i] = 1.

    x = model.predict(x={'in_qst':embedding[qst_data],
                     'in_txt':embedding[context_data],
                     'in_features':cont_features,
                     'in_tags':voc_con[con_tag],
                     'in_ents':voc_ent[ent_tag]
                    })
    beg = 0
    end = 767

    max_value = 0
    for j in range(767):
      for k in range(j, min(j+15, 767)):
        value = x[0][1][j] * x[0][0][k]
        if (value > max_value):
          max_value = value
          beg = j
          end = k
    txt = model_in[6].split()
    for i in range(beg, end + 1):
      print(txt[i], end=' ')
    print('\n_________________________________________________\n')

