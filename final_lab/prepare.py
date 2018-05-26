import numpy as np
text_len = 767
qst_len = 60


context_data = np.load('./data/context_data.npy')
qst_data = np.load('./data/qst_data.npy')
cont_features = np.load('./data/cont_features.npy')
ans_beg = np.load('./data/ans_beg.npy')
ans_end = np.load('./data/ans_end.npy')
con_tag = np.load('./data/con_tag.npy')
ent_tag = np.load('./data/ent_tag.npy')
embedding = np.load('./data/embedding.npy')
voc_ent = np.zeros((19, 19), dtype=float)
for i in range(19):
  voc_ent[i][i] = 1.
voc_con = np.zeros((50, 50), dtype=float)
for i in range(50):
  voc_con[i][i] = 1.

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

print(in_txt.shape)


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
print('tmp_sum:', tmp_sum.shape)

txt_a = Dense(hidden * 2, use_bias=False, activation='softmax', name='txt_a') (txt_result)
txt_b = Dense(hidden * 2, use_bias=False, activation='softmax', name='txt_b') (txt_result)

bb = Lambda(lambda x1: K.batch_dot(x1[0], x1[1])) ([txt_a, tmp_sum])
ee = Lambda(lambda x1: K.batch_dot(x1[0], x1[1])) ([txt_b, tmp_sum])

begin = Activation('softmax', name='ans_beg') (bb)
end = Activation('softmax', name='ans_end') (ee)

print('end:', end)

model = Model(inputs=[input_txt, input_qst, input_features, input_tags, input_ent], outputs=[begin, end])

model.compile(
    loss='categorical_crossentropy',
    optimizer='RMSprop',
    metrics=['accuracy']
)

model.summary()

