import numpy as np
context_test = np.load('./data/context_test.npy')
cont_features_test = np.load('./data/cont_features_test.npy')
con_test = np.load('./data/con_test.npy')
ent_test = np.load('./data/ent_test.npy')
qst_test = np.load('./data/qst_test.npy')
embedding = np.load('./data/embedding.npy')
test_begin = np.load('./data/test_begin.npy')
test_end = np.load('./data/test_end.npy')

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

model = Model(inputs=[input_txt, input_qst, input_features, input_tags, input_ent], outputs=[begin, end])

model.compile(
    loss='categorical_crossentropy',
    optimizer='RMSprop',
    metrics=['accuracy']
)

#model.summary()

model.load_weights('./models/last-v-3.h5')

def f1_score(num = 1000):
  w = model.predict(x={'in_qst':embedding[qst_test[0:num]],
                 'in_txt':embedding[context_test[0:num]],
                 'in_features':cont_features_test[0:num],
                 'in_tags':voc_con[con_test[0:num]],
                 'in_ents':voc_ent[ent_test[0:num]]
                })

  S = 0

  for i in range(num):
      max_value = 0
      beg = 0
      end = 767
    
      prob_begin = w[0][i]
      prob_end = w[1][i]
    
      for j in range(767):
          for k in range(j, min(j+15, 767)):
              value = prob_begin[j] * prob_end[k]
              if (value > max_value):
                  max_value = value
                  beg = j
                  end = k
    
      intersection = min(end, test_end[i]) - max(beg, test_begin[i]) + 1
      pr = intersection / (end - beg + 1)
      rec = intersection / (test_end[i] - test_begin[i] + 1)
      if pr < 0:
        pr = 0
      if rec < 0:
        rec = 0
      if (pr + rec > 0.0000001):
        S += 2 * pr * rec / (pr + rec) / num
  print(S)

f1_score()


