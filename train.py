#coding:utf-8
import os
os.environ['TF_KERAS'] = '1'
import numpy as np
from load_data import train_df, test_df
from bert4keras.tokenizers import Tokenizer
from bert4keras.models import build_transformer_model
from bert4keras.optimizers import Adam, extend_with_piecewise_linear_lr
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.snippets import open
import tensorflow as tf
from tensorflow.keras.layers import Lambda, Dense

trans={'game':0,'fashion':1,'houseliving':2}
num_classes = 3
maxlen = 512
batch_size = 32
config_path = '../albert_small_zh_google/albert_config_small_google.json'
checkpoint_path = '../albert_small_zh_google/albert_model.ckpt'
dict_path = '../albert_small_zh_google/vocab.txt'

# config_path = 'chinese_L-12_H-768_A-12/bert_config.json'
# checkpoint_path = 'chinese_L-12_H-768_A-12/bert_model.ckpt'
# dict_path = 'chinese_L-12_H-768_A-12/vocab.txt'

tokenizer = Tokenizer(dict_path, do_lower_case=True)

def tran(label):
    return trans.get(label)
class data_generator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, (label, text) in self.sample(random):
            token_ids, segment_ids = tokenizer.encode(text, max_length=maxlen)
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append([tran(label)])
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels)
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []

bert = build_transformer_model(
    config_path=config_path,
    checkpoint_path=checkpoint_path,
    model='albert',
    return_keras_model=False,
)

output = Lambda(lambda x: x[:, 0], name='CLS-token')(bert.model.output)
output = Dense(
    units=num_classes,
    activation='softmax',
    kernel_initializer=bert.initializer
)(output)

model = tf.keras.models.Model(bert.model.input, output)
# model.summary()

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

model.load_weights('../weights/model_weights')

train_generator = data_generator(train_df, batch_size)
test_generator = data_generator(test_df, batch_size)

#预测
# while True:
#     sentence=input("请输入句子")
#     if sentence=="end":
#         break
#     predict_token_ids, predict_segment_ids = tokenizer.encode(sentence, max_length=maxlen)
#     predict = model.predict([np.asarray([predict_token_ids]), np.asarray([predict_segment_ids])]).argmax(axis=1)
#     print(list(trans.keys())[int(predict)])

# predict_token_ids,predict_segment_ids=tokenizer.encode(sentence,max_length=maxlen)
# predict = model.predict([np.asarray([predict_token_ids]),np.asarray([predict_segment_ids])]).argmax(axis=1)
# print(list(trans.keys())[int(predict)])

#训练模型
def evaluate(data):
    total, right = 0., 0.
    for x_true, y_true in data:
        # print(x_true)
        y_pred = model.predict(x_true).argmax(axis=1)
        y_true = y_true[:, 0]
        total += len(y_true)
        right += (y_true == y_pred).sum()
    return right / total


class Evaluator(tf.keras.callbacks.Callback):

    def on_epoch_end(self, epoch, logs=None):
        test_acc = evaluate(test_generator)
        print(
            u' test_acc: %.5f\n' %
            (test_acc)
        )

evaluator=Evaluator()

model.fit_generator(
    train_generator.forfit(),
    steps_per_epoch=len(train_generator),
    epochs=1,
    # validation_data=test_generator.forfit()
    callbacks=[evaluator]
)

model.save_weights('../weights/model_weights')







