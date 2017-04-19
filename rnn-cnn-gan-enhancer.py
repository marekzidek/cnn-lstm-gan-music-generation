from keras.models import Sequential, Model
from keras.layers import Dense, Input
from keras.layers import Reshape
from keras.layers.core import Activation, Flatten, Dropout, Lambda
from keras.layers.convolutional import Convolution1D, Convolution2D, ZeroPadding1D
from keras.layers.pooling import GlobalMaxPooling1D, AveragePooling1D, MaxPooling1D
from keras.layers.wrappers import TimeDistributed, Bidirectional
from keras.layers.recurrent import GRU, LSTM, SimpleRNN
from keras.regularizers import l2, l1, l1_l2
from keras import backend
from keras.layers.advanced_activations import ELU, PReLU, LeakyReLU
from keras.layers import Layer
import keras.layers
import keras
import itertools
import sys
import random
import theano.tensor as T
import random
import os
import tensorflow as tf

from keras.optimizers import SGD
import numpy as np
import argparse
import math
from midi_to_matrix import *
import train as trainLoadMusic

np.random.seed(26041994)

generating_size = 50*8 # not recommended to change here or via --song_length as a change is also needed in the model
                       # that generates latent music
note_span_with_ligatures = 156 # will be used as a "magic number constant" in convolution filters

def generator_model():

    input_song = Input(shape=(generating_size,156))
    amplified = keras.layers.core.Lambda(lambda x:x * 12 - 6)(input_song)

    forget = LSTM(250, return_sequences=True)(input_song)
    forget = keras.layers.local.LocallyConnected1D(156,1,activation='sigmoid', kernel_initializer='zeros', bias_initializer=keras.initializers.Constant(-6.0))(forget)

    conservativity_sum_0 = Lambda(lambda x: backend.sum(x, axis=1), output_shape=lambda s: (s[0], s[2]))(forget)
    conservativity_sum_0 = Lambda(lambda x: backend.sum(x[:,::2], axis=1), output_shape=lambda s: (s[0],1))(conservativity_sum_0)
    tresh = keras.layers.advanced_activations.ThresholdedReLU(theta=400.0)(conservativity_sum_0)

    # this will be zero if treshold was not found,
    tresh = Lambda(lambda x:backend.log(x+1))(tresh)

    # penalty the forget gate for forgetting too much, loged to avoid vanishing gradient for next activation
    forget = Lambda(lambda x:x-tresh)(forget)

    forget = keras.layers.core.Lambda(lambda x:(5.5*x))(forget)
    forget = Activation('sigmoid')(forget)

    # multiply to be able to outvote residual 6* connection    
    forget = keras.layers.core.Lambda(lambda x:-(12*x))(forget)


    add = LSTM(250,kernel_initializer='zeros', return_sequences=True)(input_song)
    add = keras.layers.local.LocallyConnected1D(156,1,activation='sigmoid', kernel_initializer='zeros', bias_initializer=keras.initializers.Constant(-6.0))(add)
    conservativity_sum_1 = Lambda(lambda x: backend.sum(x, axis=1), output_shape=lambda s: (s[0], s[2]))(add)
    conservativity_sum_1 = Lambda(lambda x: backend.sum(x[:,::2], axis=1), output_shape=lambda s: (s[0],1))(conservativity_sum_1)

    treshold = keras.layers.advanced_activations.ThresholdedReLU(theta=60.0)(conservativity_sum_1)

    # this will be zero if treshold was not found, logarithmed to avoid vanishing problem with the succeding add gate
    treshold = Lambda(lambda x:backend.log(x+1))(treshold)

    # penalty the forget gate for forgetting too much
    add = Lambda(lambda x:x-treshold)(add)
    add = keras.layers.core.Lambda(lambda x:(5.5*x))(add)
    add = Activation('sigmoid')(add)

    # multiply to be able to outvote residual 6* connection    
    add = keras.layers.core.Lambda(lambda x:(x*12))(add)
    
    residual = keras.layers.merge([amplified, forget, add], mode='sum')
    residual = Activation('sigmoid')(residual)

    # mask the ligatures/articulations to not be learned, where the corresponding note is not played
    mask_for_articulation = keras.layers.advanced_activations.ThresholdedReLU(theta=0.5)(residual)
    mask_for_articulation = Lambda(lambda x:x[:,:,1::2], output_shape=lambda s: (s[0], s[1], 78))(mask_for_articulation)
    play =  Lambda(lambda x:x[:,:,::2], output_shape=lambda s: (s[0], s[1], 78))(residual)
    reshaped_play = Reshape((generating_size,78,1))(play)
    arti =  Lambda(lambda x:x[:,:,1::2], output_shape=lambda s: (s[0], s[1], 78))(residual)
    articulate = Lambda(lambda x: x * mask_for_articulation)(arti)
    reshaped_articulate = Reshape((generating_size,78,1))(articulate)
    final = keras.layers.concatenate([reshaped_play, reshaped_articulate])
    final = Reshape((generating_size,156))(final)

    model = Model(inputs=input_song, outputs=final)
    return model

def discriminator_model():

     ## check joint
    
    input_song = Input(shape=(generating_size,156))
    
    joint = Reshape((generating_size,156,1))(input_song)
    joint = TimeDistributed(Convolution1D(filters=20,kernel_size=8, padding='valid', strides=2))(joint) #tercie (3 pultony = sleduji 4 noty) * ligatura(proto 2 strides)
    #39
    joint = Activation(LeakyReLU(0.3))(joint)
    joint = TimeDistributed(Convolution1D(filters=40,kernel_size=3, padding='valid', strides=1))(joint) # velka tercie 4 pultony a cista kvarta 5 pultonu
    #38
    joint = Activation(LeakyReLU(0.3))(joint)
    joint = TimeDistributed(Convolution1D(filters=200,kernel_size=3, padding='valid', strides=1))(joint) # kvinty 7 pultonu = sleduji 8 not
    #17
    joint = Activation(LeakyReLU(0.3))(joint)
    joint = TimeDistributed(MaxPooling1D(2))(joint) # chci dominantni akord

    joint = TimeDistributed(Convolution1D(filters=300,kernel_size=3, padding='valid', strides=1))(joint)
    #5
    print joint.shape
    joint = Activation(LeakyReLU(0.3))(joint)
    joint = TimeDistributed(MaxPooling1D(2))(joint)
    print joint.shape
    joint = TimeDistributed(Convolution1D(filters=400,kernel_size=3, padding='valid', strides=2))(joint)
    #5
    print joint.shape
    joint = Activation(LeakyReLU(0.3))(joint)
    # (gen_size, 66, 20)
    cross_joint = Reshape((generating_size,7*400))(joint)
    joint = TimeDistributed(Dense(50))(cross_joint)
    joint = Flatten()(joint)
    joint = Dropout(0.5)(joint)
    joint = Dense(1, kernel_regularizer=keras.regularizers.l2(0.1))(joint)
    joint = Activation(LeakyReLU(0.3))(joint)

    ## check rhythm
    
    rhythm = ZeroPadding1D(4)(input_song) # 4 on both sides, so locally connecteds kernel will be 9 (bc. they don't supp 'same' yet)
    rhythm = Convolution1D(filters=20*20, kernel_size=24, strides=16, padding='valid')(rhythm)
    rhythm = Activation(LeakyReLU(0.3))(rhythm)
    rhythm = Reshape((generating_size/16, 20, 20))(rhythm)
    rhythm = TimeDistributed(keras.layers.local.LocallyConnected1D(filters=100, kernel_size=9, padding='valid'))(rhythm)
    rhythm = Activation(LeakyReLU(0.3))(rhythm)
    rhythm = TimeDistributed(Dense(50))(rhythm)
    rhythm = Flatten()(rhythm)
    rhythm = Dropout(0.5)(rhythm)
    rhythm = Dense(1, kernel_regularizer=keras.regularizers.l2(0.1))(rhythm)
    rhythm = Activation(LeakyReLU(0.3))(rhythm)

    ## check structure

    structure = Reshape((generating_size,156,1))(input_song)
    structure = TimeDistributed(Convolution1D(filters=16,kernel_size=8, padding='same', strides=4))(structure) #tercie*ligatura
    # 78
    structure = Activation(LeakyReLU(0.3))(structure)
    structure = TimeDistributed(Convolution1D(filters=32,kernel_size=2, padding='valid', strides=2))(structure) #kvinty
    structure = TimeDistributed(MaxPooling1D(2))(structure)
    structure = Reshape((generating_size,9*32))(structure)
    structure = Convolution1D(80,2)(structure)
    structure = Activation(LeakyReLU(0.3))(structure)
    structure = Convolution1D(120,2, dilation_rate=2)(structure)
    structure = Activation(LeakyReLU(0.3))(structure)
    structure = Convolution1D(160,2, dilation_rate=4)(structure)
    structure = Activation(LeakyReLU(0.3))(structure)
    structure = Convolution1D(200,2, dilation_rate=8)(structure)
    structure = Activation(LeakyReLU(0.3))(structure)
    structure = TimeDistributed(Dense(50))(structure)
    structure = Dropout(0.5)(structure)
    structure = Flatten()(structure)
    structure = Dense(1, kernel_regularizer=keras.regularizers.l2(0.1))(structure)
    structure = Activation(LeakyReLU(0.3))(structure)

    ## check consistency

    differences = Reshape((generating_size,156,1))(input_song)
    differences = TimeDistributed(Convolution1D(filters=1,kernel_size=2, padding='same', strides=2))(differences) #tercie*ligatura
    # 78
    differences = Activation(LeakyReLU(0.3))(differences)
    differences = Reshape((generating_size,78))(differences)
    differences = Convolution1D(150,2)(differences)
    differences = SimpleRNN(200,return_sequences=True)(differences)
    differences = TimeDistributed(Dense(1,kernel_regularizer=keras.regularizers.l2(0.1)))(differences)
    differences = Activation(LeakyReLU(0.3))(differences)
    differences = Flatten()(differences)
    differences = Dropout(0.5)(differences)
    differences = Dense(1, kernel_regularizer=keras.regularizers.l2(0.1))(differences)
    differences = Activation(LeakyReLU(0.3))(differences)

    continuity = GRU(150,return_sequences=True)(cross_joint)
    continuity = Activation(LeakyReLU(0.3))(continuity)
    continuity = TimeDistributed(Dense(1,kernel_regularizer=keras.regularizers.l2(0.1)))(continuity)
    continuity = Flatten()(continuity)
    continuity = Dropout(0.5)(continuity)
    continuity = Dense(1, kernel_regularizer=keras.regularizers.l2(0.1))(continuity)
    continuity =  Activation(LeakyReLU(0.3))(continuity)

    final = keras.layers.concatenate([joint, rhythm, structure, continuity, differences])
    final = Dropout(0.35)(final)
    final = Dense(1)(final)
    #final = Activation('sigmoid')(final) # Do not use in Wasserstein GAN (also use mean_squared_error)

    model = Model(inputs=input_song, outputs=final)
    return model

def generator_with_discriminator_model(generator, discriminator):
    
    model = Sequential()
    model.add(generator)

    discriminator.trainable = False
    #model.add(Reshape((generating_size*156,1)))
    model.add(discriminator)
    
    return model
    
    
def createBatches(music_list, SONG_LENGTH, BATCH_SIZE):

    if len(music_list) == 0:
        print "None music in music or lstm_outputs folder"
        sys.exit()

    batch_random_indices = range(len(music_list))
    random.shuffle(batch_random_indices)

    train_X = []
    for i in range(len(music_list)/BATCH_SIZE):
        batch = []
        for j in range(BATCH_SIZE):
            prepart = music_list[batch_random_indices.pop(0)]
            part = []
            
            for timestep in prepart:
                merged = list(itertools.chain.from_iterable(timestep))
                part.append(merged)

            batch.append(part)
        batch = np.array(batch)
        train_X.append(batch)

    return np.array(train_X)
        

## tahle metoda je odporna. potom prepsat at tam neni 2x skoro uplne to stejny
def generate_from_midis(path_memory, path_train):

    batch_size = 3
    if len(os.listdir(path_memory)) < len(os.listdir(path_train)):
        memory_music_names = os.listdir(path_memory)
        memory_music_names = [ midi for midi in memory_music_names if midi[-4:] in ('.mid', '.MID')]
        while 1:
            random_indices = range(len(os.listdir(path_train)))
            random.shuffle(random_indices)
            random_pos = 0
            batch_pos = 0
            for name in [midi for midi in os.listdir(path_train)if midi[-4:] in ('.mid', '.MID')]:
                x_memory = midiToMatrix(os.path.join(path_memory, memory_music_names[random_indices[random_pos] % len(memory_music_names)]))[:generating_size]
                x_train = midiToMatrix(os.path.join(path_train, name))[:generating_size]
                random_pos += 1
                if len(x_memory) < generating_size or len(x_train) < generating_size:
                    continue
                x_memory = np.array(x_memory).reshape((1,generating_size, note_span_with_ligatures))
                x_train = np.array(x_train).reshape((1,generating_size, note_span_with_ligatures))

                x = np.concatenate([x_memory,x_train])
                
                if batch_pos % batch_size == 0:
                    batch_x = x
                    batch_y = np.array([-0.5,0.5]).reshape((2,1))
                else:
                    batch_x = np.concatenate([batch_x, x])
                    batch_y = np.concatenate([batch_y, np.array([-0.5,0.5]).reshape((2,1))])
                
                batch_pos += 1
                if batch_pos % batch_size == 0:
                    yield (batch_x, batch_y)
    else:
        train_music_names = os.listdir(path_train)
        train_music_names = [ midi for midi in train_music_names if midi[-4:] in ('.mid', '.MID')]

        while 1:
            random_indices = range(len(os.listdir(path_memory)))
            random.shuffle(random_indices)
            random_pos = 0
            batch_pos = 0
            for name in [midi for midi in os.listdir(path_memory)if midi[-4:] in ('.mid', '.MID')]:
                x_memory = midiToMatrix(os.path.join(path_memory, name))[:generating_size]
                x_train = midiToMatrix(os.path.join(path_train, train_music_names[random_indices[random_pos] % len(train_music_names)]))[:generating_size]
                random_pos += 1
                if len(x_memory) < generating_size or len(x_train) < generating_size:
                    continue
                x_memory = np.array(x_memory).reshape((1,generating_size, note_span_with_ligatures))
                x_train = np.array(x_train).reshape((1,generating_size, note_span_with_ligatures))

                x = np.concatenate([x_memory,x_train])
                
                if batch_pos % batch_size == 0:
                    batch_x = x
                    batch_y = np.array([-0.5,0.5]).reshape((2,1))
                else:
                    batch_x = np.concatenate([batch_x, x])
                    batch_y = np.concatenate([batch_y, np.array([-0.5,0.5]).reshape((2,1))])
                
                batch_pos += 1
                if batch_pos % batch_size == 0:
                    yield (batch_x, batch_y)


def train(BATCH_SIZE, SONG_LENGTH, EPOCH):

    sys.setrecursionlimit(100000)

    discriminator = discriminator_model()
    print "loading latent music"
    latent_music = trainLoadMusic.loadMusic("lstm_outputs", SONG_LENGTH)
    latent_music = latent_music.values()

    print "creating discriminator"
    discriminator = discriminator_model()
    print "created discriminator"
    generator = generator_model()
    generator_with_discriminator = generator_with_discriminator_model(generator, discriminator)

    #d_optim = keras.optimizers.RMSprop(lr=0.0001, rho=0.9, epsilon=1e-08, decay=0.0)
    d_optim = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    g_optim = keras.optimizers.RMSprop(lr=0.0001, rho=0.9, epsilon=1e-08, decay=0.0, clipnorm=0.5, clipvalue=0.5)
    #g_optim = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    #g_optim = keras.optimizers.Adadelta(lr=1.0,rho=0.95,epsilon=1e-8,decay=0.0)

    generator.compile(loss='binary_crossentropy', optimizer="adam")
    generator_with_discriminator.compile(
        loss='mean_squared_error', optimizer=g_optim)


    discriminator.trainable = True
    discriminator.compile(loss='mean_squared_error', optimizer=d_optim)

    
    print "loading weighsts"
    #generator.load_weights('generator_pretrain')

    print "loaded weights"

    latent_batches = createBatches(latent_music, SONG_LENGTH, BATCH_SIZE)  
    
    # save memory for discriminator to not forget about this being FAKE
    for j in range(len(latent_batches)):
        generated_music = generator.predict_on_batch(latent_batches[j])
        for k in range(BATCH_SIZE):
            song = generated_music[k].reshape((SONG_LENGTH,note_span_with_ligatures/2,2))
            song = generate_from_probabilities(song)
            matrixToMidi(song,'discriminator_memory/after pre epochs {} example {}'.format(j,k))
    
    print "loading disc weights"
    #discriminator.load_weights('discriminator 2')
    print "loaded disc weights"
    discriminator.load_weights('discriminator 2')
    
    discriminator.fit_generator(generate_from_midis("discriminator_memory", "music"), steps_per_epoch=40, epochs=7)
    print discriminator.layers[-1].get_weights()
    
    for epoch in range(1, 100):
        
        #if epoch % 8 == 0:
        #    generator.load_weights('generator_identity')

        #latent_batches = createBatches(latent_music, SONG_LENGTH, BATCH_SIZE)
        
        for indexer in xrange(3*len(latent_batches)):
            # latent_batches (batch_size, song_length, notes)
            # generated_music = generator.predict_on_batch(latent_batches[indexer % len(latent_batches)])
            
            discriminator.trainable = False
            latent_batches = createBatches(latent_music, SONG_LENGTH, BATCH_SIZE)

            
            
            for i in range(3):

                what_to_train = [0.5 for j in range(BATCH_SIZE)]
                g_loss = generator_with_discriminator.train_on_batch(
                    latent_batches[indexer % len(latent_batches)], np.array([what_to_train]).reshape((BATCH_SIZE,1)))
                evalu = generator_with_discriminator.predict_on_batch(latent_batches[indexer % len(latent_batches)])
                print evalu
                print("epoch %d, batch %d gen_loss : %f" % (epoch, indexer % len(latent_batches), g_loss))
                generated_music = generator.predict_on_batch(latent_batches[indexer % len(latent_batches)])
                song_0 = generated_music[0].reshape((SONG_LENGTH,note_span_with_ligatures/2,2))
                song_0 = generate_from_probabilities(song_0)
                matrixToMidi(song_0,'outputs/test {} {}'.format(i, indexer))

            
            if indexer % 10 == 0:
                song_0 = generated_music[0].reshape((SONG_LENGTH,note_span_with_ligatures/2,2))
                song_0 = generate_from_probabilities(song_0, conservativity=1.0)
                matrixToMidi(song_0,'outputs/after {} epochs {} example 0'.format(epoch, indexer))

                if indexer % len(latent_batches) == 0:
                    generator.save_weights('generator {} indexer {} '.format(epoch, indexer), True)
        
        generator.save_weights('generator {}'.format(epoch), True)

        
        folder = 'discriminator_memory'
        for the_file in os.listdir(folder):
            file_path = os.path.join(folder, the_file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                #elif os.path.isdir(file_path): shutil.rmtree(file_path)
            except Exception as e:
                print(e)

        print "saving generated songs for discriminator"
        # save memory for discriminator to not forget about this being FAKE
        for i in range(len(latent_batches)):
            generated_music = generator.predict_on_batch(latent_batches[i])
            for j in range(BATCH_SIZE):
                song = generated_music[j].reshape((SONG_LENGTH,note_span_with_ligatures/2,2))
                song = generate_from_probabilities(song)
                matrixToMidi(song,'discriminator_memory/after {} epochs, batch {} example {}'.format(epoch, i, j))
        
        print "saved all songs"

        discriminator.trainable = True
        
        discriminator.fit_generator(generate_from_midis("discriminator_memory", "music"), steps_per_epoch=5, epochs=2)
        
        discriminator.save_weights('discriminator {}'.format(epoch))


def generate(SONG_LENGTH, nb):
    
    generator = generator_model()
    generator.compile(loss='binary_crossentropy', optimizer="SGD")
    generator.load_weights('generator')
    
    print "loading_latent_music"
    latent_music = trainLoadMusic.loadMusic("lstm_outputs", SONG_LENGTH)

    for i in range(nb):

        latent = random.choice(latent_music)

        song = generator.predict(latent, verbose=1)

        song = song.reshape((SONG_LENGTH,note_span_with_ligatures/2,2))
        song_0 = generate_from_probabilities(song_0)
        matrixToMidi(song_0,'outputs/example {}'.format(i))

def generate_from_probabilities(song, conservativity=1):

    for i in range(len(song)):
        for j in range(len(song[i])):
            song[i][j][0] = np.random.sample(1) < song[i][j][0] * conservativity
            song[i][j][1] = np.random.sample(1) < song[i][j][1] * conservativity
    return song

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="generate")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--song_length", type=int, default=50*8)
    parser.add_argument("--nb", type=int, default=1)
    parser.add_argument("--epoch", type=int, default=1)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    generating_size = args.song_length
    if args.mode == "train":
        train(BATCH_SIZE=args.batch_size, SONG_LENGTH=args.song_length, EPOCH=args.epoch)
    elif args.mode == "generate":
        generate(SONG_LENGTH=args.song_length,nb=args.n)
