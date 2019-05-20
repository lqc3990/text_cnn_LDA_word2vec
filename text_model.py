#encoding:utf-8
import  tensorflow as tf
import os
import numpy as np
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
class TextConfig():
#your path
    yourpath_dir='drive/text_cnn_LDA_word2vec'
#CNN config
    embedding_size=100     #dimension of word embedding
    vocab_size=100247        #number of vocabulary,see in the vector_word.txt(first word)
    seq_length=200         #max length of sentence
    num_classes=3          #number of labels
    num_filters=256        #number of convolution kernel
    kernel_size=7          #size of convolution kernel
    hidden_dim=128         #number of fully_connected layer units
    keep_prob=0.5          #droppout
    lr= 1e-3              #learning rate
    lr_decay= 0.9          #learning rate decay
    clip= 5.0              #gradient clipping threshold
    LDA_size=10
    num_epochs=10          #epochs
    batch_size= 128         #batch_size
    print_per_batch =100   #print result
#orignial data
    train_filename=os.path.join(yourpath_dir, 'data/train.txt')  #train data
    test_filename=os.path.join(yourpath_dir, 'data/test.txt'  )  #test data
    val_filename=os.path.join(yourpath_dir, 'data/val.txt'  )      #validation data
#cleanning data(for LDA)
    train_LDA_data=os.path.join(yourpath_dir, 'cleanning_data/fenci_train_nolabel.txt')  #train data
    test_LDA_data=os.path.join(yourpath_dir, 'cleanning_data/fenci_test_nolabel.txt')    #test data
    val_LDA_data=os.path.join(yourpath_dir, 'cleanning_data/fenci_val_nolabel.txt')      #validation data
#cleanning data(for Word2vec or CNN )
    train_w2v_data=os.path.join(yourpath_dir, 'cleanning_data/fenci_train_label.txt')  
    test_w2v_data=os.path.join(yourpath_dir, 'cleanning_data/fenci_test_label.txt'   )
    val_w2v_data=os.path.join(yourpath_dir, 'cleanning_data/fenci_test_label.txt')  
#word2vec path
    vocab_filename=os.path.join(yourpath_dir, 'word2vec_pretrain/vocab.txt')        #vocabulary
    vector_word_filename=os.path.join(yourpath_dir, 'word2vec_pretrain/vector_word.txt')  #vector_word trained by word2vec
    vector_word_npz=os.path.join(yourpath_dir, 'word2vec_pretrain/vector_word.npz')   # save vector_word to numpy file

    pre_trianing = None   #use vector_char trained by word2vec
#LDA path
    LDA_size=200
    pre_trianing_LDAtrain = os.path.join(yourpath_dir, 'LDA_pretrain/LDA_trainvec.npy') 
    pre_trianing_LDAval = os.path.join(yourpath_dir, 'LDA_pretrain/LDA_valvec.npy')
    pre_trianing_LDAtest = os.path.join(yourpath_dir, 'LDA_pretrain/LDA_testvec.npy')
    #LDA_model_path=os.path.join(yourpath_dir, 'lda_model/' )
#CNN save  
    #save_dir=os.path.join(yourpath_dir, 'checkpoints/textcnn')
    #tensorboard_dir=os.path.join(yourpath_dir, 'tensorboard/textcnn')
#stopwords
    stpwrdpath=os.path.join(yourpath_dir, 'data/stop_words.txt')

class TextCNN(object):

    def __init__(self,config):

        self.config=config

        self.input_x=tf.placeholder(tf.int32,shape=[None,self.config.seq_length],name='input_x')
        self.x_train_topic=tf.placeholder(tf.float32,shape=[None,config.LDA_size],name='x_train_topic')
        self.input_y=tf.placeholder(tf.float32,shape=[None,self.config.num_classes],name='input_y')
        self.flag=tf.placeholder(tf.float32,name='flag')
        self.keep_prob=tf.placeholder(tf.float32,name='dropout')
       
        self.global_step = tf.Variable(0, trainable=False, name='global_step')

        self.cnn()
    def cnn(self):
        with tf.device('/gpu:0'):
            self.embedding = tf.get_variable("embeddings", shape=[self.config.vocab_size, self.config.embedding_size],
                                             initializer=tf.constant_initializer(self.config.pre_trianing))
            embedding_inputs=tf.nn.embedding_lookup(self.embedding,self.input_x)
            if self.flag is not None:
               embedding_inputs=embedding_inputs
            else:
               embedding_inputs=np.mean(embedding_inputs, axis=1)
               embedding_inputs=np.concatenate([x_train_topic, embedding_inputs], axis=1)

        with tf.name_scope('cnn'):
            conv= tf.layers.conv1d(embedding_inputs, self.config.num_filters, self.config.kernel_size, name='conv')
            outputs= tf.reduce_max(conv, reduction_indices=[1], name='gmp')

        with tf.name_scope('fc'):
            fc=tf.layers.dense(outputs,self.config.hidden_dim,name='fc1')
            fc = tf.nn.dropout(fc, self.keep_prob)
            fc=tf.nn.relu(fc)

        with tf.name_scope('logits'):
            self.logits = tf.layers.dense(fc, self.config.num_classes, name='logits')
            self.prob = tf.nn.softmax(self.logits)
            self.y_pred_cls = tf.argmax(tf.nn.softmax(self.logits), 1)

        with tf.name_scope('loss'):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
            self.loss = tf.reduce_mean(cross_entropy)

        with tf.name_scope('optimizer'):
            self.learning_rates = tf.train.exponential_decay(self.config.lr, global_step=self.global_step,decay_steps=1000, decay_rate=0.95, staircase=True)

            optimizer = tf.train.AdamOptimizer(self.learning_rates)
            gradients, variables = zip(*optimizer.compute_gradients(self.loss))
            gradients, _ = tf.clip_by_global_norm(gradients, self.config.clip)
            self.optim = optimizer.apply_gradients(zip(gradients, variables), global_step=self.global_step)


        with tf.name_scope('accuracy'):
            correct_pred=tf.equal(tf.argmax(self.input_y,1),self.y_pred_cls)
            self.acc=tf.reduce_mean(tf.cast(correct_pred,tf.float32))