#coding:utf-8
import os
os.environ['TF_KERAS'] = '1'
import numpy as np
from bert4keras.tokenizers import Tokenizer
from bert4keras.models import build_transformer_model
from bert4keras.optimizers import Adam, extend_with_piecewise_linear_lr
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.snippets import open
import tensorflow as tf
from tensorflow.keras.layers import Lambda, Dense
from tensorflow.keras.models import Model

trans={'O':0,'B-PER':1,'I-PER':2,'B-LOC':3,'I-LOC':4,'B-ORG':5,'I-ORG':6}
trans_id={0:'O',1:'B-PER',2:'I-PER',3:'B-LOC',4:'I-LOC',5:'B-ORG',6:'I-ORG'}
num_classes = 7
maxlen = 512
batch_size = 32
epochs = 10
bert_layers = 6
learing_rate = 1e-5  # bert_layers越小，学习率应该要越大

config_path = '../albert_small_zh_google/albert_config_small_google.json'
checkpoint_path = '../albert_small_zh_google/albert_model.ckpt'
dict_path = '../albert_small_zh_google/vocab.txt'

# config_path = 'chinese_L-12_H-768_A-12/bert_config.json'
# checkpoint_path = 'chinese_L-12_H-768_A-12/bert_model.ckpt'
# dict_path = 'chinese_L-12_H-768_A-12/vocab.txt'
def load_data(filename):
    D = []
    with open(filename, encoding='utf-8') as f:
        f = f.read()
        for l in f.split('\n\n'):
            if not l:
                continue
            d, last_flag = [], ''
            for c in l.split('\n'):
                char, this_flag = c.split(' ')
                # if this_flag == 'O' and last_flag == 'O':
                #     d[-1][0] += char
                # elif this_flag == 'O' and last_flag != 'O':
                #     d.append([char, 'O'])
                # elif this_flag[:1] == 'B':
                #     d.append([char, this_flag[2:]])
                # else:
                #     d[-1][0] += char
                # last_flag = this_flag
                d.append([char,this_flag])
            D.append(d)
    return D


# 标注数据
train_data = load_data('../china-people-daily-ner-corpus/example.train')
valid_data = load_data('../china-people-daily-ner-corpus/example.dev')
test_data = load_data('../china-people-daily-ner-corpus/example.test')


tokenizer = Tokenizer(dict_path, do_lower_case=True)

def tran(label):
    return trans.get(label)
class data_generator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, item in self.sample(random):
            token_ids, labels = [tokenizer._token_start_id], [0]
            for w, l in item:
                w_token_ids = tokenizer.encode(w)[0][1:-1]
                if len(token_ids) + len(w_token_ids) < maxlen:
                    token_ids += w_token_ids
                    labels.append(tran(l))
                else:
                    break
            token_ids += [tokenizer._token_end_id]
            labels += [0]
            segment_ids = [0] * len(token_ids)
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append(labels)
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels)
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []


model = build_transformer_model(
    config_path,
    checkpoint_path,
    model='albert',
)

output_layer = 'Transformer-FeedForward-Norm'
output = model.get_layer(output_layer).get_output_at(bert_layers - 1)
output = Dense(num_classes,activation='softmax')(output)

model = Model(model.input, output)
model.summary()
# 派生为带分段线性学习率的优化器。
# 其中name参数可选，但最好填入，以区分不同的派生优化器。
AdamLR = extend_with_piecewise_linear_lr(Adam, name='AdamLR')

model.compile(
    loss='sparse_categorical_crossentropy',
    # optimizer=Adam(1e-5),  # 用足够小的学习率
    optimizer=AdamLR(learning_rate=1e-4, lr_schedule={
        1000: 1,
        2000: 0.1
    }),
    metrics=['accuracy'],
)

train_generator = data_generator(train_data, batch_size)
test_generator = data_generator(test_data, batch_size)

model.load_weights('../weights_ner/model_weights')
# 预测
while True:
    sentence=input("请输入句子")
    if sentence=="end":
        break
    predict_token_ids, predict_segment_ids = tokenizer.encode(sentence, max_length=maxlen)
    predict = model.predict([np.asarray([predict_token_ids]), np.asarray([predict_segment_ids])])
    predict=np.squeeze(predict)
    predict=predict[1:-1]
    result=[]
    for item in predict:
        result.append(trans_id.get(np.argmax(item)))
    print(result)


#训练模型
# def evaluate(data):
#     total, right = 0., 0.
#     for x_true, y_true in data:
#         # print(x_true)
#         y_pred = model.predict(x_true).argmax(axis=1)
#         y_true = y_true[:, 0]
#         total += len(y_true)
#         right += (y_true == y_pred).sum()
#     return right / total
#
#
# class Evaluator(tf.keras.callbacks.Callback):
#
#     def on_epoch_end(self, epoch, logs=None):
#         test_acc = evaluate(test_generator)
#         print(
#             u' test_acc: %.5f\n' %
#             (test_acc)
#         )
#
# evaluator=Evaluator()

# model.fit_generator(
#     train_generator.forfit(),
#     steps_per_epoch=len(train_generator),
#     epochs=1,
#     validation_data=test_generator.forfit(),
#     validation_steps=1
# )

# model.save_weights('../weights_ner/model_weights')







