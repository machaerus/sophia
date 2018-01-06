import pandas as pd
import re
import sys
import os
import numpy as np
import tensorflow as tf
import pickle
import time

print("Parsing Bible...")

book_list = os.listdir("kjv")
book_list = sorted(book_list)

Bible = []

for bookname in book_list:
    
    book_num = bookname[:2]
    
    with open("kjv/" + bookname) as f:
        f_input = f.read()
    spl = re.split(r"(\{[0-9]+\:[0-9]+\})", f_input)
#     book = []
    book_title = spl.pop(0).strip()
    
    
    while(spl):
        
        num = spl.pop(0)
        text = spl.pop(0)
    #     book.verses.append((num.strip(), verse.strip()))
        chapter, paragraph = num.strip()[1:-1].split(":")

        if int(chapter) > 99: 
            chapter = str(chapter)
        elif int(chapter) > 9:
            chapter = "0"+str(chapter)
        else:
            chapter = "00"+str(chapter)

        if int(paragraph) > 99: 
            paragraph = str(paragraph)
        elif int(paragraph) > 9:
            paragraph = "0"+str(paragraph)
        else:
            paragraph = "00"+str(paragraph)

        Bible.append({
            "book": book_num,
            "chapter": chapter,
            "paragraph": paragraph,
            "short_name": bookname[3:-4],
            "text": re.sub("\s+", " ", text.strip())
        })
        
Bible = pd.DataFrame(Bible)

NT = Bible[(Bible.book.astype(int) >= 1) & (Bible.book.astype(int) <= 66)]

NT_concat = ""
for p in NT.text:
    NT_concat += (p + " ")

with open("arabian_nights.txt") as f:
    an = f.read()
an = re.sub("\.+",".",an)
an = re.sub("\s+"," ",an)
an = re.sub("\"","",an)
NT_concat += an

with open("gilgamesh.txt") as f:
    gil = f.read()
gil = re.sub("\.+",".", gil)
gil = re.sub("\s+"," ",gil)
NT_concat += gil


corpus_raw = NT_concat

print("Preparing NN...")

def create_lookup_tables(text):
    """
    Create lookup tables for vocab
    :param text: The GOT text split into words
    :return: A tuple of dicts (vocab_to_int, int_to_vocab)
    """
    vocab = set(text)
    int_to_vocab = {key: word for key, word in enumerate(vocab)}
    vocab_to_int = {word: key for key, word in enumerate(vocab)}
    return vocab_to_int, int_to_vocab

def token_lookup():
    """
    Generate a dict to map punctuation into a token
    :return: dictionary mapping puncuation to token
    """
    return {
        '.': '||period||',
        ',': '||comma||',
        '"': '||quotes||',
        ';': '||semicolon||',
        '!': '||exclamation-mark||',
        '?': '||question-mark||',
        '(': '||left-parentheses||',
        ')': '||right-parentheses||',
        '--': '||emm-dash||',
        '\n': '||return||'
        
    }



token_dict = token_lookup()
for token, replacement in token_dict.items():
    corpus_raw = corpus_raw.replace(token, ' {} '.format(replacement))
corpus_raw = corpus_raw.lower()
corpus_raw = corpus_raw.split()

vocab_to_int, int_to_vocab = create_lookup_tables(corpus_raw)
corpus_int = [vocab_to_int[word] for word in corpus_raw]
pickle.dump((corpus_int, vocab_to_int, int_to_vocab, token_dict), open('preprocess.p', 'wb'))

def get_batches(int_text, batch_size, seq_length):
    """
    Return batches of input and target data
    :param int_text: text with words replaced by their ids
    :param batch_size: the size that each batch of data should be
    :param seq_length: the length of each sequence
    :return: batches of data as a numpy array
    """
    words_per_batch = batch_size * seq_length
    num_batches = len(int_text)//words_per_batch
    int_text = int_text[:num_batches*words_per_batch]
    y = np.array(int_text[1:] + [int_text[0]])
    x = np.array(int_text)
    
    x_batches = np.split(x.reshape(batch_size, -1), num_batches, axis=1)
    y_batches = np.split(y.reshape(batch_size, -1), num_batches, axis=1)
    
    batch_data = list(zip(x_batches, y_batches))
    
    return np.array(batch_data)

num_epochs = 10000
batch_size = 512
rnn_size = 512
num_layers = 3
keep_prob = 0.7
embed_dim = 512
seq_length = 30
learning_rate = 0.001
save_dir = './save'


train_graph = tf.Graph()
with train_graph.as_default():    
    
    # Initialize input placeholders
    input_text = tf.placeholder(tf.int32, [None, None], name='input')
    targets = tf.placeholder(tf.int32, [None, None], name='targets')
    lr = tf.placeholder(tf.float32, name='learning_rate')
    
    # Calculate text attributes
    vocab_size = len(int_to_vocab)
    input_text_shape = tf.shape(input_text)
    
    # Build the RNN cell
    lstm = tf.contrib.rnn.BasicLSTMCell(num_units=rnn_size)
    drop_cell = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=keep_prob)
    cell = tf.contrib.rnn.MultiRNNCell([drop_cell] * num_layers)
    
    # Set the initial state
    initial_state = cell.zero_state(input_text_shape[0], tf.float32)
    initial_state = tf.identity(initial_state, name='initial_state')
    
    # Create word embedding as input to RNN
    embed = tf.contrib.layers.embed_sequence(input_text, vocab_size, embed_dim)
    
    # Build RNN
    outputs, final_state = tf.nn.dynamic_rnn(cell, embed, dtype=tf.float32)
    final_state = tf.identity(final_state, name='final_state')
    
    # Take RNN output and make logits
    logits = tf.contrib.layers.fully_connected(outputs, vocab_size, activation_fn=None)
    
    # Calculate the probability of generating each word
    probs = tf.nn.softmax(logits, name='probs')
    
    # Define loss function
    cost = tf.contrib.seq2seq.sequence_loss(
        logits,
        targets,
        tf.ones([input_text_shape[0], input_text_shape[1]])
    )
    
    # Learning rate optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate)
    
    # Gradient clipping to avoid exploding gradients
    gradients = optimizer.compute_gradients(cost)
    capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients if grad is not None]
    train_op = optimizer.apply_gradients(capped_gradients)
    

pickle.dump((seq_length, save_dir), open('params.p', 'wb'))
batches = get_batches(corpus_int, batch_size, seq_length)
num_batches = len(batches)
start_time = time.time()

print("Training...")

with tf.Session(graph=train_graph) as sess:
    sess.run(tf.global_variables_initializer())
    
    for epoch in range(num_epochs):
        state = sess.run(initial_state, {input_text: batches[0][0]})
        
        for batch_index, (x, y) in enumerate(batches):
            feed_dict = {
                input_text: x,
                targets: y,
                initial_state: state,
                lr: learning_rate
            }
            train_loss, state, _ = sess.run([cost, final_state, train_op], feed_dict)
            
        time_elapsed = time.time() - start_time
        print('Epoch {:>3} Batch {:>4}/{}   train_loss = {:.3f}   time_elapsed = {:.3f}   time_remaining = {:.0f}'.format(
            epoch + 1,
            batch_index + 1,
            len(batches),
            train_loss,
            time_elapsed,
            ((num_batches * num_epochs)/((epoch + 1) * (batch_index + 1))) * time_elapsed - time_elapsed))

        # save model every 10 epochs
        if epoch % 10 == 0:
            saver = tf.train.Saver()
            saver.save(sess, save_dir)
            print('Model Trained and Saved')
            

