from numpy import array
from numpy import argmax
from keras.callbacks import Callback
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import Input, Dense, Flatten, LSTM, RepeatVector, BatchNormalization,\
TimeDistributed, Embedding, concatenate, GlobalMaxPooling2D, GRU, Dropout, Activation
import matplotlib.pyplot as plt
import os
import numpy as np

lr = 1e-3

EMBEDDING_DIM = 100
MAX_NUM_WORDS = 20000

# define the captioning model
def define_model_last_5_out(vocab_size, max_length, emb_mat):
    # feature extractor (encoder)
    inputs1 = Input(shape=(7, 7, 512))
    fe1 = Flatten()(inputs1)
    return commonLayers(fe1,vocab_size,max_length,inputs1, emb_mat)
    
def define_model_last_6_out(vocab_size, max_length,emb_mat):
    # feature extractor (encoder)
    inputs1 = Input(shape=(14, 14, 512))
    fe1 = Flatten()(inputs1)
    return commonLayers(fe1,vocab_size,max_length,inputs1,emb_mat)

def define_model_last_3_out(vocab_size, max_length,emb_mat,bn=True,drop=True):
    # feature extractor (encoder)
    inputs1 = Input(shape=(4096,))
    return commonLayers(inputs1,vocab_size,max_length,inputs1,emb_mat,bn,drop)
    
def commonLayers(fe1,vocab_size,max_length,inputs1,emb_mat,bn=True,drop=True):
    #image encoding
    fe2 = Dense(256)(fe1)
    if bn:
        fe2 = BatchNormalization()(fe2)
    fe2 = Activation('relu')(fe2)
    if drop:
        fe2 = Dropout(0.25)(fe2)
    fe3 = RepeatVector(max_length)(fe2)
    
    #partial caption encoding
    # embedding
    inputs2 = Input(shape=(max_length,))
    emb2 = Embedding(vocab_size,
                     EMBEDDING_DIM,
                     weights=[emb_mat],
                     trainable=False,
                     mask_zero=True)(inputs2)
    emb3 = GRU(512, return_sequences=True,unroll=True)(emb2)
    emb4 = TimeDistributed(Dense(256))(emb3)
    if bn:
        emb4 = BatchNormalization()(emb4)
    emb4 = Activation('relu')(emb4)
    if drop:
        emb4 = Dropout(0.25)(emb4)
    
    # language model (decoder)
    # merge inputs
    merged = concatenate([fe3, emb4])
    if bn and drop:
        merged = GRU(512,unroll=True,return_sequences=True)(merged)
    merged = GRU(512,unroll=True)(merged)
    lm3 = Dense(512)(merged)
    if bn:
        lm3 = BatchNormalization()(lm3)
    lm3 = Activation('relu')(lm3)
    if drop:
        lm3 = Dropout(0.25)(lm3)
    outputs = Dense(vocab_size, activation='softmax')(lm3)
    # tie it together [image, seq] [word]
    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.optimizer.lr = lr
    print(model.summary())
    return model

def get_embedding_matrix(tokenizer):
    
    word_index = tokenizer.word_index
    
    embeddings_index = {}
    with open(os.path.join('glove', 'glove.6B.{0}d.txt'.format(EMBEDDING_DIM)), encoding='utf8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    
    num_words = min(MAX_NUM_WORDS, len(word_index) + 1)
    embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
    for word, i in word_index.items():
        if i >= MAX_NUM_WORDS:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix


# create sequences of images, input sequences and output words for an image
def create_sequences(tokenizer, desc, image, max_length):
    Ximages, XSeq, y = list(), list(),list()
    vocab_size = len(tokenizer.word_index) + 1
    # integer encode the description
    seq = tokenizer.texts_to_sequences([desc])[0]
    # split one sequence into multiple X,y pairs
    for i in range(1, len(seq)):
        # select
        in_seq, out_seq = seq[:i], seq[i]
        # pad input sequence
        in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
        # encode output sequence
        out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
        # store
        Ximages.append(image)
        XSeq.append(in_seq)
        y.append(out_seq)
    # Ximages, XSeq, y = array(Ximages), array(XSeq), array(y)
    return [Ximages, XSeq, y]

# data generator, intended to be used in a call to model.fit_generator()
def data_generator(descriptions, features, tokenizer, max_length, n_step):
    # loop until we finish training
    while 1:
        # loop over photo identifiers in the dataset
        keys = list(descriptions.keys())
        for i in range(0, len(keys), n_step):
            Ximages, XSeq, y = list(), list(),list()
            for j in range(i, min(len(keys), i+n_step)):
                image_id = keys[j]
                # retrieve photo feature input
                image = features[image_id][0]
                # retrieve text input
                desc = descriptions[image_id]
                # generate input-output pairs
                in_img, in_seq, out_word = create_sequences(tokenizer, desc, image, max_length)
                for k in range(len(in_img)):
                    Ximages.append(in_img[k])
                    XSeq.append(in_seq[k])
                    y.append(out_word[k])
            # yield this batch of samples to the model
            yield [[array(Ximages), array(XSeq)], array(y)]

def show_curve_train(history):
    plt.plot(history.history['acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train'], loc='upper left')
    plt.show()

class WeightsSaver(Callback):
    def __init__(self, model,name,step=1):
        self.model = model
        self.name = name
        self.epoch = 0
        self.step = step
        
    def on_epoch_end(self, epoch, logs={}):
        if self.epoch%self.step == 0 or self.epoch == 199:
            self.model.save_weights(self.name+'Epoch'+str(self.epoch)+'.h5')
        self.epoch+=1
