import gensim
import os
import numpy as np
import keras
import pprint
from keras.layers import Input, Convolution2D, MaxPooling2D, Dense, Dropout, Flatten, AveragePooling2D
from keras.models import Sequential, Model
from keras.layers.merge import concatenate
    
def sentence2array(sentence_arr, length, model):
    ret = []
    count = 0
    for word in sentence_arr:
        if count >= length:
            break
        try:
            ret.append(model.word_vec(word))
        except KeyError:
            ret.append(np.random.uniform(-1,1,300))
        count += 1
    for i in xrange(len(sentence_arr), length):
        ret.append(np.zeros(300))
    return ret
        
def prepare_data(pos_dir, neg_dir, model):
    maxlen = 30
    neg_sentences = []
    with open(neg_dir, mode='r') as infile:
        for line in infile:
            neg_sentences.append([i for i in line.split(' ')])
            # maxlen = max(maxlen, len(line.split(' ')))
    pos_sentences = []
    with open(pos_dir, mode='r') as infile:
        for line in infile:
            pos_sentences.append([i for i in line.split(' ')])
            # maxlen = max(maxlen, len(line.split(' ')))
    
    neg_data = []
    pos_data = []
    for sentence in neg_sentences:
        neg_data.append(sentence2array(sentence, maxlen, model))        
    for sentence in pos_sentences:
        pos_data.append(sentence2array(sentence, maxlen, model))
        
    pos_arr = np.asarray(pos_data)
    neg_arr = np.asarray(neg_data)    

    pos_labels = np.zeros(shape=len(pos_arr), dtype=np.int8)
    neg_labels = np.ones(shape=len(neg_arr), dtype=np.int8)
    
    x = np.vstack((pos_arr, neg_arr))
    y = np.hstack((pos_labels, neg_labels))
    x = x.reshape(len(x), maxlen, 300, 1)
    
    return x, keras.utils.to_categorical(y, num_classes=2)

def prepare_TREC(dir, model):
    maxlen = 11
    sentences = []
    labels = []
    with open(dir, mode='r') as infile:
        for line in infile:
            sentences.append([i for i in line.split(' ')])
            labels.append(sentences[-1][0])
            sentences[-1].pop()
            
    label_dict = {'DESC' : 0, 'ENTY' : 1, 'ABBR' : 2, 'HUM' : 3, 'LOC' : 4, 'NUM' : 5}
    y = []
    for label in labels:
        y.append(label_dict[label.split(':')[0]])
    
    data = []
    for sentence in sentences:
        data.append(sentence2array(sentence, maxlen, model))
    
    
    x = np.asarray(data)
    x = x.reshape(len(x), maxlen, 300, 1)
    return x, keras.utils.to_categorical(y, num_classes=len(label_dict))
    
# Load Google's pre-trained Word2Vec model.
word2vec_model = gensim.models.KeyedVectors.load_word2vec_format('./model/GoogleNews-vectors-negative300.bin', binary=True)
# x, y = prepare_data('./data/rt-polaritydata/rt-polarity.pos', './data/rt-polaritydata/rt-polarity.neg', word2vec_model)
x, y = prepare_TREC('./data/questions.txt', word2vec_model)
# x_test, y_test = prepare_TREC('./data/questions_test.txt', word2vec_model)


# for i in (3,4,5) :
    # for j in (3,4,5) :
        # print (i, j)
        # kernel_size = 3
        # feature_maps = 50
        # num_classes = len(y[0])
        # num_epochs = 50

        # inp = Input(shape=x[0].shape)
        # conv0 = Convolution2D(feature_maps, (i, i), padding='same', activation='relu')(inp)
        # pool0 = MaxPooling2D(pool_size=(2, 2))(conv0)
        # conv1 = Convolution2D(feature_maps, (j, j), padding='same', activation='relu')(pool0)
        # pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        # drop = Dropout(0.50)(pool1)
        # flat = Flatten()(drop)
        # out = Dense(num_classes, activation='softmax')(flat)

        # model = Model(inputs=inp, outputs=out)

        # model.compile(loss='categorical_crossentropy',
                      # optimizer='adam',
                      # metrics=['accuracy'])

        # model.fit(x, y, batch_size=50, epochs=num_epochs,
                  # verbose=2, validation_split=0.1)
                      



feature_maps = 50
num_classes = len(y[0])
num_epochs = 50

inp = Input(shape=x[0].shape)
conv0 = Convolution2D(feature_maps, (7, 7), padding='same', activation='relu')(inp)
# pool0 = MaxPooling2D(pool_size=(3, 3))(conv0)
pool0 = conv0

conv00 = Convolution2D(feature_maps, (3, 3), padding='same', activation='relu')(pool0)
conv01 = Convolution2D(feature_maps, (4, 4), padding='same', activation='relu')(conv00)
conv02 = Convolution2D(feature_maps * 2, (5, 5), padding='same', activation='relu')(conv01)

conv10 = Convolution2D(feature_maps * 2, (3, 3), padding='same', activation='relu')(pool0)

pool20 = AveragePooling2D((3,3), strides=(1,1), padding='same')(pool0)
conv20 = Convolution2D(feature_maps * 2, (4, 4), padding='same', activation='relu')(pool20)

concat = concatenate([conv02, conv10, conv20], axis = 1)

pool2 = MaxPooling2D(pool_size=(2, 2))(concat)
drop = Dropout(0.50)(pool2)
flat = Flatten()(drop)
out = Dense(num_classes, activation='softmax')(flat)

model = Model(inputs=inp, outputs=out)

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(x, y, batch_size=50, epochs=num_epochs,
          verbose=1, validation_split=0.1)
          
# x_test, y_test = prepare_TREC('./data/questions_test.txt', word2vec_model)
# print(model.evaluate(x_test, y_test, verbose=1))  

# loss_and_metrics = model.evaluate(x_test, y_test, batch_size=64, verbose=1)

# classes = model.predict(x_test, batch_size=128)
