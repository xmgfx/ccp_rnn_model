# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.contrib.layers import l2_regularizer
from module import placeholder, super_params
from module import local_network
from data.loader import Loader

"""
对历史记录的处理网络
输出:
lstm_output LSTM网络输出
mcard_action_2s_before_output 打牌的之前两次动作压缩编码
"""
record_placeholder = placeholder.RecordPlaceholder()
record_network = local_network.RecordNetwork(record_placeholder)
lstm_output, mcard_action_2s_before_output = record_network.output()

"""
手牌向量的信息压缩网络
输出:
deck_vec_eocode_network_output 压缩向量
"""
deck_vec_placeholder = placeholder.DeckVecPlaceholder()
deck_vec_eocode_network = local_network.DeckVecEncodeNetwork(deck_vec_placeholder)
deck_vec_eocode_network_output = deck_vec_eocode_network.output()

"""
手牌数量转向量网络
输出:
num_deck_encode_network_output 手牌数量向量
"""
num_deck_placeholder = placeholder.NumDeckPlaceholder()
num_deck_encode_network = local_network.NumDeckEmbedNetwork(num_deck_placeholder)
num_deck_encode_network_output = num_deck_encode_network.output()

"""
手牌可执行动作压缩网络
输出:
deck_mcard_action_encode_network_output: 压缩网络
"""
deck_mcard_action_placeholder = placeholder.DeckMcardActionPlaceholder()
deck_mcard_action_encode_network = local_network.DeckMcardActionEncodeNetwork(deck_mcard_action_placeholder)
deck_mcard_action_encode_network_output = deck_mcard_action_encode_network.output()

"""
对上面所有得到的信息拼接
"""
total_message = tf.concat(values=[lstm_output,
                                  mcard_action_2s_before_output,
                                  deck_vec_eocode_network_output,
                                  num_deck_encode_network_output,
                                  deck_mcard_action_encode_network_output], axis=-1)
"""
利用上面信息预测玩家下一次动作
"""
pred_next_mcard_action = tf.layers.dense(inputs=total_message, units=super_params.num_mcard_action)
pred_next_mcard_action += deck_mcard_action_placeholder.inf_hands_action  # 加掩模

"""
真实动作
"""
true_label_placeholder = placeholder.TrueLabelPlaceholder()

"""
为参数增加正则化
正则化误差
"""
l2_reg = l2_regularizer(scale=super_params.l2_regularizer_scale)
l2_loss = tf.reduce_sum([l2_reg(_) for _ in tf.trainable_variables()])

"""
模型预测误差
"""
model_pred_loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred_next_mcard_action,
                                                                               labels=true_label_placeholder.next_mcard_action))
"""
总误差
"""
total_loss = l2_loss + model_pred_loss

"""
指数衰减学习率
"""
control_placeholder = placeholder.ControlPlaceholder()

learning_rate = tf.train.exponential_decay(learning_rate=super_params.learning_rate_base,
                                           decay_rate=super_params.learning_rate_decay,
                                           global_step=control_placeholder.global_step,
                                           decay_steps=super_params.learning_rate_decay_step)

optim = tf.train.AdamOptimizer(learning_rate=super_params.learning_rate_standard)
grads = optim.compute_gradients(loss=total_loss)

for i, (g, v) in enumerate(grads):
    if g is not None:
        grads[i] = (tf.clip_by_norm(g, 5), v)
train_op = optim.apply_gradients(grads)

pred_action = tf.argmax(pred_next_mcard_action, axis=-1)
accuracy = tf.reduce_mean(tf.cast(tf.equal(pred_action, true_label_placeholder.next_mcard_action), dtype=tf.float32))

saver = tf.train.Saver()

path = "../data/sample100w"
loader = Loader(path)
i = 0
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver.restore(sess=sess, save_path="model/rnnmodel.ckpt")
    for epoch in range(super_params.num_epoch):
        for batch_data in loader.read_batch(batch_size=super_params.batch_size):
            feed_dict = {
                record_placeholder.record_mcard_action: batch_data["mcard_action_record"],
                record_placeholder.record_ktype_action: batch_data["ktype_action_record"],
                record_placeholder.record_klen_action: batch_data["klen_action_record"],
                record_placeholder.mcard_action_2s_before: batch_data["mcard_record_2s_before"],
                deck_vec_placeholder.hands_vec: batch_data["hands_vec"],
                deck_vec_placeholder.hidden_cards_vec: batch_data["hidden_cards_vec"],
                deck_vec_placeholder.output_cards_vec: batch_data["output_cards_vec"],
                num_deck_placeholder.num_hands: batch_data["num_hands"],
                num_deck_placeholder.num_output_cards: batch_data["num_output_cards"],
                num_deck_placeholder.num_hidden_cards: batch_data["num_hidden_cards"],
                num_deck_placeholder.num_up_hands: batch_data["num_up_hands"],
                num_deck_placeholder.num_down_hands: batch_data["num_down_hands"],
                deck_mcard_action_placeholder.hands_action: batch_data["hands_action_vec"],
                deck_mcard_action_placeholder.hidden_cards_action: batch_data["hidden_cards_action_vec"],
                deck_mcard_action_placeholder.inf_hands_action: batch_data["inf_hands_action_vec"],
                deck_mcard_action_placeholder.inf_hidden_cards_action: batch_data["inf_hidden_cards_action_vec"],
                true_label_placeholder.next_mcard_action: batch_data["next_mcard_action_label"]
            }
            _, acc, tl = sess.run([train_op, accuracy, total_loss], feed_dict=feed_dict)
            print("accuracy:", acc, "total_loss:", tl)
            i += 1
            if i % 10000 == 100:
                saver.save(sess=sess, save_path="model/rnnmodel.ckpt")
