import tensorflow as tf
import re

BATCH_SIZE = 128
MAX_WORDS_IN_REVIEW = 100  # Maximum length of a review to consider
EMBEDDING_SIZE = 50  # Dimensions for each word vector

stop_words = set({'ourselves', 'hers', 'between', 'yourself', 'again',
                  'there', 'about', 'once', 'during', 'out', 'very', 'having',
                  'with', 'they', 'own', 'an', 'be', 'some', 'for', 'do', 'its',
                  'yours', 'such', 'into', 'of', 'most', 'itself', 'other',
                  'off', 'is', 's', 'am', 'or', 'who', 'as', 'from', 'him',
                  'each', 'the', 'themselves', 'below', 'are', 'we',
                  'these', 'your', 'his', 'through', 'don', 'me', 'were',
                  'her', 'more', 'himself', 'this', 'down', 'should', 'our',
                  'their', 'while', 'above', 'both', 'up', 'to', 'ours', 'had',
                  'she', 'all', 'no', 'when', 'at', 'any', 'before', 'them',
                  'same', 'and', 'been', 'have', 'in', 'will', 'on', 'does',
                  'yourselves', 'then', 'that', 'because', 'what', 'over',
                  'why', 'so', 'can', 'did', 'not', 'now', 'under', 'he', 'you',
                  'herself', 'has', 'just', 'where', 'too', 'only', 'myself',
                  'which', 'those', 'i', 'after', 'few', 'whom', 't', 'being',
                  'if', 'theirs', 'my', 'against', 'a', 'by', 'doing', 'it',
                  'how', 'further', 'was', 'here', 'than'})

punctuations = re.compile('[^a-zA-Z ]')

def preprocess(review):
    """
    Apply preprocessing to a single review. You can do anything here that is manipulation
    at a string level, e.g.
        - removing stop words
        - stripping/adding punctuation
        - changing case
        - word find/replace
    RETURN: the preprocessed review in string form.
    """
    review = punctuations.sub('', review)
    review = review.lower()
    words = review.split(' ')
    processed_review = []
    for word in words:
        if word not in stop_words:
            processed_review.append(word)
    return processed_review


def define_graph():
    """
    Implement your model here. You will need to define placeholders, for the input and labels,
    Note that the input is not strings of words, but the strings after the embedding lookup
    has been applied (i.e. arrays of floats).

    In all cases this code will be called by an unaltered runner.py. You should read this
    file and ensure your code here is compatible.

    Consult the assignment specification for details of which parts of the TF API are
    permitted for use in this function.

    You must return, in the following order, the placeholders/tensors for;
    RETURNS: input, labels, optimizer, accuracy and loss
    """
    num_class = 2
    safe_learning_rate = 0.001

    # input_data, labels, dropout_keep_prob
    input_data = tf.placeholder(tf.float32, shape=[BATCH_SIZE, MAX_WORDS_IN_REVIEW, EMBEDDING_SIZE],  name="input_data")
    labels = tf.placeholder(tf.float32,  shape=[BATCH_SIZE, num_class], name="labels")
    dropout_keep_prob = tf.placeholder_with_default(0.8, shape=(), name='dropout_keep_prob')

    # lstm instance
    lstm_forward_1 = tf.contrib.rnn.BasicLSTMCell(num_units = 200, forget_bias = 1.0, state_is_tuple = True)
    # lstm_forward_2 = tf.contrib.rnn.BasicLSTMCell(num_units = 200, forget_bias = 1.0, state_is_tuple = True)
    # lstm_forward = tf.contrib.rnn.MultiRNNCell(cells = [lstm_forward_1, lstm_forward_2])

    lstm_backward_1 = tf.contrib.rnn.BasicLSTMCell(num_units=200, forget_bias=1.0, state_is_tuple=True)
    # lstm_backward_2 = tf.contrib.rnn.BasicLSTMCell(num_units=200, forget_bias=1.0, state_is_tuple=True)
    # lstm_backward = tf.contrib.rnn.MultiRNNCell(cells=[lstm_backward_1, lstm_backward_2])

    #output
    output_fw = tf.contrib.rnn.DropoutWrapper(lstm_forward_1, output_keep_prob=dropout_keep_prob)
    output_bw = tf.contrib.rnn.DropoutWrapper(lstm_backward_1, output_keep_prob=dropout_keep_prob)

    (outputs_fw, output_bw), states = tf.nn.bidirectional_dynamic_rnn(cell_fw=output_fw,
                                                      cell_bw=output_bw,
                                                      inputs=input_data,
                                                      dtype=tf.float32)

    final_output = outputs_fw[:,-1,:] + output_bw[:,-1,:]
    logits = tf.layers.dense(final_output, num_class)

    #loss, optimizer, accuracy
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels), name='loss')
    optimizer = tf.train.AdamOptimizer(learning_rate=safe_learning_rate).minimize(loss)

    correct_prediction = tf.equal(tf.argmax(tf.nn.softmax(logits), axis=1), tf.argmax(labels, axis=1))
    Accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')

    return input_data, labels, dropout_keep_prob, optimizer, Accuracy, loss
