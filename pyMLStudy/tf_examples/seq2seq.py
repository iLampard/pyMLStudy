import numpy as np
import tensorflow as tf
from tensorflow.python.layers.core import Dense


# https://github.com/NELSONZHAO/zhihu/blob/master/basic_seq2seq/Seq2seq_char.ipynb

def load_data():
    with open('letters_source.txt', 'r', encoding='utf-8') as f:
        source_data = f.read()

    with open('letters_target.txt', 'r', encoding='utf-8') as f:
        target_data = f.read()
    return source_data, target_data


def extract_character_vocab(data):
    """
    构造映射表
    """
    special_words = ['<PAD>', '<UNK>', '<GO>', '<EOS>']
    set_words = list(set([char for line in data.split('\n') for char in line]))
    # 这里要把四个特殊字符添加进词典
    int_to_vocab = {idx: word for idx, word in enumerate(special_words + set_words)}
    vocat_to_int = {word: idx for idx, word in int_to_vocab.items()}

    return int_to_vocab, vocat_to_int


def input_layer():
    input = tf.placeholder(tf.int32, [None, None], name='input')
    target = tf.placeholder(tf.int32, [None, None], name='target')
    learning_rate = tf.placeholder(tf.float32, name='learning_rate')

    target_sequence_len = tf.placeholder(tf.int32, (None,), name='target_sequence_len')
    max_target_sequence_len = tf.reduce_max(target_sequence_len, name='max_target_len')
    source_seq_len = tf.placeholder(tf.int32, name='source_seq_len')

    return input, target, learning_rate, target_sequence_len, max_target_sequence_len, source_seq_len


def encoder_layer(input_data, rnn_size, num_layers, source_seq_len, source_vocab_size, encoding_embedding_size):
    # [batch_size, source_vocab_size(time_step), embed_dim]
    encoder_embed_input = tf.contrib.layers.embed_sequence(input_data, source_vocab_size, encoding_embedding_size)

    def get_lstm_cell(rnn_size):
        cell = tf.contrib.rnn.LSTMCell(rnn_size, initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
        return cell
    cell = tf.contrib.rnn.MultiRNNCell([get_lstm_cell(rnn_size) for _ in range(num_layers)])

    encoder_output, encoder_state = tf.nn.dynamic_rnn(cell, encoder_embed_input, sequence_length=source_seq_len,
                                                      dtype=tf.float32)

    return encoder_output, encoder_state


def process_decoder_input(data, vocab_to_int, batch_size):
    """
    补充<GO>，并移除最后一个字符
    """
    ending = tf.strided_slice(data, [0,0])



def main():
    source_data, target_data = load_data()

    # 构造映射表
    source_int_to_letter, source_letter_to_int = extract_character_vocab(source_data)
    target_int_to_letter, target_letter_to_int = extract_character_vocab(target_data)

    # 对字母进行转换
    source_int = [[source_letter_to_int.get(char, source_letter_to_int['<UNK']) for char in line] for line in
                  source_data.split('\n')]
    target_int = [[target_letter_to_int.get(char, target_letter_to_int['<UNK']) for char in line] + [
        target_letter_to_int['<EOS>']] for line in target_data.split('\n')]

    return


if __name__ == '__main__':
    main()
