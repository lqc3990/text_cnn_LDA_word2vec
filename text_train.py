#encoding:utf-8
from __future__ import print_function
from text_model import *
from loader import *
from sklearn import metrics
import sys
import os
import time
from datetime import timedelta
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
import umap

 


def feed_data(x_batch, y_batch, x_topic,keep_prob,valflag):
    feed_dict = {
        model.input_x: x_batch,
        model.input_y: y_batch,
        model.x_train_topic: x_topic,
        model.keep_prob:keep_prob,
        model.flag:valflag
    }
    return feed_dict

def train():
    print("Configuring TensorBoard and Saver...")
    
    save_dir = 'drive/text_cnn_LDA_word2vec/checkpoints/textcnn'#change drive/  to  yourpath
    tensorboard_dir = 'drive/text_cnn_LDA_word2vec/tensorboard/textcnn'#change drive/  to  yourpath
    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, 'best_validation')

    print("Loading training and validation data...")
    start_time = time.time()
    x_train, y_train,x_train_topic = process_file(config.train_w2v_data, word_to_id, cat_to_id, config.seq_length)
    x_val, y_val,x_val_topic = process_file(config.val_w2v_data, word_to_id, cat_to_id, config.seq_length)
    print("Time cost: %.3f seconds...\n" % (time.time() - start_time))
    tf.summary.scalar("loss", model.loss)
    tf.summary.scalar("accuracy", model.acc)
    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter(tensorboard_dir)
    saver = tf.train.Saver()
    
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    writer.add_graph(session.graph)

    print('Training and evaluating...')
    best_val_accuracy = 0
    last_improved = 0  # record global_step at best_val_accuracy
    require_improvement = 100000  # break training if not having improvement over 1000 iter
    flag=False
    val_flag=False

    for epoch in range(config.num_epochs):
        batch_train = batch_iter(x_train, y_train,pre_trianing_LDAtrain,config.batch_size)
        start = time.time()
        print('Epoch:', epoch + 1)
        for x_batch, y_batch ,x_topic_batch in batch_train:
            feed_dict = feed_data(x_batch, y_batch, x_topic_batch,config.keep_prob,val_flag)
            _, global_step, train_summaries, train_loss, train_accuracy = session.run([model.optim, model.global_step,
                                                                                    merged_summary, model.loss,
                                                                                    model.acc], feed_dict=feed_dict)
            if global_step % config.print_per_batch == 0:
                end = time.time()
                feed_dict = feed_data(x_val, y_val,pre_trianing_LDAval, 1.0,val_flag)
                val_summaries, val_loss, val_accuracy,learning_rates = session.run([merged_summary, model.loss, model.acc,model.learning_rates ],
                                                                 feed_dict=feed_dict)
                writer.add_summary(val_summaries, global_step)
                # If improved, save the model
                if val_accuracy > best_val_accuracy:
                    saver.save(session, save_path)
                    best_val_accuracy = val_accuracy
                    last_improved=global_step
                    improved_str = '*'
                else:
                    improved_str = ''
                print("step: {},train loss: {:.3f}, train accuracy: {:.3f}, val loss: {:.3f}, val accuracy: {:.3f}, learning_rates: {:>6.2} {}\n".format(
                        global_step, train_loss, train_accuracy, val_loss, val_accuracy,  learning_rates,improved_str))                
               		
                start = time.time()

            if global_step - last_improved > require_improvement:
                print("No optimization over 1000 steps, stop training")
                flag = True
                break
        if flag:
            break
        config.lr *= config.lr_decay

if __name__ == '__main__':
    print('Configuring CNN model...')
    config = TextConfig()
    print(config.__dict__)
    #middatafolder =config.LDA_model_path
    pre_trianing_LDAtrain=[] 
    pre_trianing_LDAval=[]
    filenames = [config.train_filename, config.test_filename, config.val_filename]
    if not os.path.exists(config.vocab_filename):
        build_vocab(filenames, config.vocab_filename, config.vocab_size)

    #read vocab and categories
    categories,cat_to_id = read_category()
    words,word_to_id = read_vocab(config.vocab_filename)
    config.vocab_size = len(words)

    # trans vector file to numpy file
    if not os.path.exists(config.vector_word_npz):
        export_word2vec_vectors(word_to_id, config.vector_word_filename, config.vector_word_npz)
    config.pre_trianing = get_training_word2vec_vectors(config.vector_word_npz)

    pre_trianing_LDAtrain= get_training_LDA_vectors(config.pre_trianing_LDAtrain)

    pre_trianing_LDAval = get_training_LDA_vectors(config.pre_trianing_LDAval)
    model = TextCNN(config)
    train()
