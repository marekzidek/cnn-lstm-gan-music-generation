from keras.models import Sequential, Model
from keras.layers import Dense, Input
from keras.layers import Reshape
from keras.layers.core import Activation, Flatten, Dropout, Lambda
from keras.layers.convolutional import Convolution1D, Convolution2D, ZeroPadding1D
from keras.layers.pooling import GlobalMaxPooling1D, AveragePooling1D, MaxPooling1D
from keras.layers.wrappers import TimeDistributed, Bidirectional
from keras.layers.recurrent import GRU, LSTM, SimpleRNN
from keras import backend
from keras.layers import Layer
import keras.layers
import keras
import itertools
import sys
import random
import theano.tensor as T
import random
import os

from keras.optimizers import SGD
import numpy as np
import argparse
import math
from midi_to_matrix import *
import train as trainLoadMusic

np.random.seed(26041994)

generating_size = 50*8 # not recommended to change here or via --song_length as a change is also needed in the model
                       # generating latent music 
note_span_with_ligatures = 156 # will be used as a "magic number constant" in convolution filters

def generator_model():


    input_song = Input(shape=(generating_size,156))
    amplified = keras.layers.core.Lambda(lambda x:x * 18 - 9)(input_song)

    forget = LSTM(156, activation='sigmoid', kernel_initializer='zeros', return_sequences=True)(input_song)

    forget = keras.layers.core.Lambda(lambda x:-(18*x))(forget)

    add = LSTM(156, activation='sigmoid', kernel_initializer='zeros',return_sequences=True)(input_song)
    conservativity_sum = Lambda(lambda x: backend.sum(x, axis=1), output_shape=lambda s: (s[0], s[2]))(add)
    print conservativity_sum.shape
    conservativity_sum = Lambda(lambda x: backend.sum(x, axis=1), output_shape=lambda s: (s[0],1))(conservativity_sum)
    print conservativity_sum.shape
    ## tady ovlivnuju decko aby nevyzralo na rodice (diky tomu to vlastne funguje)
    tresholdfound = keras.layers.advanced_activations.ThresholdedReLU(theta=70.0)(conservativity_sum)
    add = Lambda(lambda x:x - tresholdfound)(add)
    add = Activation('relu')(add)
    
    add = keras.layers.core.Lambda(lambda x:(20*x))(add)
    
    #lstm = TimeDistributed(Dense(156, kernel_initializer='zeros'))(lstm)
    
    residual = keras.layers.merge([amplified, forget, add], mode='sum')
    residual = Activation('sigmoid')(residual)

    model = Model(inputs=input_song, outputs=residual)
    return model

def discriminator_model():

    ## check joint

    input_song = Input(shape=(generating_size,156))
    

    joint = Reshape((generating_size,156,1))(input_song)
    joint = TimeDistributed(Convolution1D(filters=20,kernel_size=6, padding='valid', strides=2))(joint) #tercie * ligatura(proto 2 strides)
    #39
    joint = Activation('relu')(joint)
    joint = TimeDistributed(Convolution1D(filters=50,kernel_size=3, padding='valid', strides=1))(joint) #kvinty
    #38
    joint = Activation('relu')(joint)
    joint = TimeDistributed(Convolution1D(filters=200,kernel_size=9, padding='valid', strides=2))(joint) #cela oktava (12 not)
    #17
    joint = Activation('relu')(joint)
    joint = TimeDistributed(Convolution1D(filters=300,kernel_size=3, padding='valid', strides=2))(joint)
    #5
    joint = Activation('relu')(joint)
    joint = TimeDistributed(MaxPooling1D(2))(joint)
    joint = TimeDistributed(Convolution1D(filters=400,kernel_size=3, padding='valid', strides=2))(joint)
    #5
    joint = Activation('relu')(joint)
    # (gen_size, 66, 20)
    cross_joint = Reshape((generating_size,3*400))(joint)
    joint = TimeDistributed(Dense(1))(cross_joint)
    joint = Activation('relu')(joint)
    joint = Flatten()(joint)
    #sem zkus dropout
    joint = Dropout(0.5)(joint)
    joint = Dense(3)(joint)
    joint = Activation('tanh')(joint)

    ## check rhythm
    
    rhythm = ZeroPadding1D(4)(input_song) # 4 on both sides, so locally connecteds kernel will be 9 (bc. they don't supp 'same' yet)
    rhythm = Convolution1D(filters=24*10, kernel_size=24, strides=16, padding='valid')(rhythm)
    rhythm = Activation('relu')(rhythm)
    rhythm = Reshape((generating_size/16, 24, 10))(rhythm)
    rhythm = TimeDistributed(keras.layers.local.LocallyConnected1D(filters=40, kernel_size=9, padding='valid'))(rhythm)
    rhythm = Activation('relu')(rhythm)
    rhythm = TimeDistributed(Dense(1))(rhythm)
    rhythm = Activation('relu')(rhythm)
    rhythm = Flatten()(rhythm)
    #sem zkus dropout
    rhythm = Dropout(0.5)(rhythm)
    rhythm = Dense(2)(rhythm)
    rhythm = Activation('tanh')(rhythm)

    ## check structure

    structure = Reshape((generating_size,156,1))(input_song)
    structure = TimeDistributed(Convolution1D(filters=16,kernel_size=6, padding='same', strides=4))(structure) #tercie*ligatura
    # 78
    structure = Activation('relu')(structure)
    structure = TimeDistributed(Convolution1D(filters=25,kernel_size=2, padding='valid', strides=2))(structure) #kvinty
    structure = TimeDistributed(MaxPooling1D(2))(structure)
    structure = Reshape((generating_size,9*25))(structure)
    structure = TimeDistributed(Dense(40))(structure)
    structure = Convolution1D(60,2)(structure)
    structure = Activation('relu')(structure)
    structure = Convolution1D(80,2, dilation_rate=2)(structure)
    structure = Activation('relu')(structure)
    structure = Convolution1D(100,2, dilation_rate=4)(structure)
    structure = Activation('relu')(structure)
    structure = Convolution1D(120,2, dilation_rate=8)(structure)
    structure = Activation('relu')(structure)
    structure = TimeDistributed(Dense(1))(structure)
    structure = Activation('relu')(structure)
    #sem zkus dropout
    structure = Dropout(0.5)(structure)
    structure = Flatten()(structure)
    structure = Dense(3)(structure)
    structure = Activation('tanh')(structure)

    ## check consistency

    differences = Reshape((generating_size,156,1))(input_song)
    differences = TimeDistributed(Convolution1D(filters=1,kernel_size=2, padding='same', strides=2))(differences) #tercie*ligatura
    # 78
    differences = Activation('relu')(differences)
    differences = Reshape((generating_size,78))(differences)
    differences = Convolution1D(150,2)(differences)
    differences = SimpleRNN(200,return_sequences=True)(differences)
    differences = TimeDistributed(Dense(1))(differences)
    differences = Activation('relu')(differences)
    differences = Flatten()(differences)
    differences = Dropout(0.5)(differences)
    differences = Dense(1)(differences)

    continuity = GRU(150,return_sequences=True)(cross_joint)
    continuity = Activation('relu')(continuity)
    continuity = TimeDistributed(Dense(1))(continuity)
    continuity = Flatten()(continuity)
    continuity = Dropout(0.5)(continuity)
    continuity = Dense(1)(continuity)

    final = keras.layers.concatenate([joint, rhythm, structure, continuity, continuity, continuity, differences, differences, differences])
    final = Dropout(0.35)(final)
    final = Dense(1)(final)
    final = Activation('sigmoid')(final)

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

    batch_size = 5
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
                    batch_y = np.array([0,1]).reshape((2,1))
                else:
                    batch_x = np.concatenate([batch_x, x])
                    batch_y = np.concatenate([batch_y, np.array([0,1]).reshape((2,1))])
                
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
                    batch_y = np.array([0,1]).reshape((2,1))
                else:
                    batch_x = np.concatenate([batch_x, x])
                    batch_y = np.concatenate([batch_y, np.array([0,1]).reshape((2,1))])
                
                batch_pos += 1
                if batch_pos % batch_size == 0:
                    yield (batch_x, batch_y)


def train(BATCH_SIZE, SONG_LENGTH, EPOCH):

    sys.setrecursionlimit(100000)

    discriminator = discriminator_model()
    #print "loading train_music"
    #train_music = trainLoadMusic.loadMusic("music", SONG_LENGTH)
    #train_music = train_music.values()
    print "loading latent music"
    latent_music = trainLoadMusic.loadMusic("lstm_outputs", SONG_LENGTH)
    latent_music = latent_music.values()

    print "creating discriminator"
    discriminator = discriminator_model()
    print "created discriminator"
    generator = generator_model()
    generator_with_discriminator = generator_with_discriminator_model(generator, discriminator)

    #d_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)
    #d_optim = keras.optimizers.RMSprop(lr=0.0001, rho=0.9, epsilon=1e-08, decay=0.0)
    d_optim = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    #g_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)
    g_optim = keras.optimizers.RMSprop(lr=0.0001, rho=0.9, epsilon=1e-08, decay=0.0)
    #g_optim = keras.optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    #g_optim = keras.optimizers.Adadelta(lr=1.0,rho=0.95,epsilon=1e-8,decay=0.0)

    generator.compile(loss='binary_crossentropy', optimizer="adam")
    generator_with_discriminator.compile(
        loss='binary_crossentropy', optimizer=g_optim)


    discriminator.trainable = True
    discriminator.compile(loss='binary_crossentropy', optimizer=d_optim)

    
    print "loading weighsts"
    #generator.load_weights('generator_pretrain')

    print "loaded weights"


    ## UNCOMMENT ON NEW TRAIN
    
    latent_batches = createBatches(latent_music, SONG_LENGTH, BATCH_SIZE)
    
    '''
    for i in range(len(latent_batches)/5):
        index = i % len(latent_batches)
        loss = generator.train_on_batch(latent_batches[index], latent_batches[index])
        print "pretrain loss ", loss, " epoch ", i/len(latent_batches)
        '''
    
    generator.save_weights('generator_identity')
    
    # save memory for discriminator to not forget about this being FAKE
    for j in range(len(latent_batches)):
        generated_music = generator.predict_on_batch(latent_batches[j])
        for k in range(BATCH_SIZE):
            song = generated_music[k].reshape((SONG_LENGTH,note_span_with_ligatures/2,2))
            song = generate_from_probabilities(song)
            matrixToMidi(song,'discriminator_memory/after pre epochs {} example {}'.format(j,k))
    
    #generator.load_weights("generator 3")
    print "loading disc weights"
    #discriminator.load_weights('discriminator 99')
    
    print "loaded disc weights"
    steps = len(latent_batches)*80*BATCH_SIZE
    steps = steps/4
    #discriminator.fit_generator(generate_from_midis("discriminator_memory", "music"), steps_per_epoch=100, epochs=3)
    #discriminator.fit_generator(generate_from_midis("discriminator_memory", "music"), steps_per_epoch=500, epochs=1)
    #discriminator.save_weights('discriminator first train')
    discriminator.load_weights('discriminator first train')
    discriminator.fit_generator(generate_from_midis("discriminator_memory", "music"), steps_per_epoch=50, epochs=1)
    for epoch in range(54, 100):
        
        if epoch % 4 == 0:
            generator.load_weights('generator_identity')

        latent_batches = createBatches(latent_music, SONG_LENGTH, BATCH_SIZE)
        
        for indexer in xrange(4*len(latent_batches)):
            # latent_batches (batch_size, song_length, notes)
            # generated_music = generator.predict_on_batch(latent_batches[indexer % len(latent_batches)])
            
            discriminator.trainable = False
            latent_batches = createBatches(latent_music, SONG_LENGTH, BATCH_SIZE)

            
            
            for i in xrange(1):

                #ZKUS TADY TEN FIT bude to lip fungovat s optimizerama
                ## keep surprise
                what_to_train = [random.uniform(1.0, 1.0) for i in range(BATCH_SIZE)]
                g_loss = generator_with_discriminator.train_on_batch(
                    latent_batches[indexer % len(latent_batches)], np.array([what_to_train]).reshape((BATCH_SIZE,1)))
                print("epoch %d, batch %d gen_loss : %f" % (epoch, indexer % len(latent_batches), g_loss))
            
            if indexer % 10 == 0:
                song_0 = generated_music[0].reshape((SONG_LENGTH,note_span_with_ligatures/2,2))
                song_0 = generate_from_probabilities(song_0, conservativity=0.8)
                matrixToMidi(song_0,'outputs/after {} epochs {} example 0'.format(epoch, indexer))

                if indexer % len(latent_batches) == 0:
                    generator.save_weights('generator {} indexer {} '.format(epoch, indexer), True)
        
        # generator trosku nejel
        #generator.fit(np.array(latent_batches).reshape((len(latent_batches),generating_size,156)), np.array([1]*len(latent_batches)), batch_size=2, epochs=4,verbose=1)
        generator.save_weights('generator {}'.format(epoch), True)

        

        print "saving generated songS for discriminator"
        # save memory for discriminator to not forget about this being FAKE
        for i in range(len(latent_batches)):
            generated_music = generator.predict_on_batch(latent_batches[i])
            for j in range(BATCH_SIZE):
                song = generated_music[j].reshape((SONG_LENGTH,note_span_with_ligatures/2,2))
                song = generate_from_probabilities(song)
                matrixToMidi(song,'discriminator_memory/after {} epochs, batch {} example {}'.format(epoch, i, j))
        
        print "saved all songs"

        discriminator.trainable = True
        
        for i in range(10):
            index = i % len(latent_batches)
            hard_train = discriminator.train_on_batch(generator.predict_on_batch(latent_batches[index]), np.array([np.random.uniform(0.0,0.0)]*BATCH_SIZE).reshape((BATCH_SIZE,1)))
            print "hard train loss ", hard_train, " example ", i

        steps_disc_epoch = (epoch +1) * len(latent_batches)
        discriminator.fit_generator(generate_from_midis("discriminator_memory", "music"), steps_per_epoch=20, epochs=1)

        for i in range(5):
            index = i % len(latent_batches)
            hard_train = discriminator.train_on_batch(generator.predict_on_batch(latent_batches[index]), np.array([np.random.uniform(0.0,0.0)]*BATCH_SIZE).reshape((BATCH_SIZE,1)))
            print "hard train loss ", hard_train, " example ", i

        discriminator.fit_generator(generate_from_midis("discriminator_memory", "music"), steps_per_epoch=10, epochs=1)

        
        discriminator.save_weights('discriminator {}'.format(epoch))


def generate(SONG_LENGTH, nb):
    
    generator = generator_model()
    generator.compile(loss='binary_crossentropy', optimizer="SGD")
    generator.load_weights('generator')
    
    print "loading_latent_music"
    latent_music = trainLoadMusic.loadMusic("lstm_outputs", SONG_LENGTH)

    for i in range(nb):

        noise = random.choice(latent_music)

        # pozor predict bere v tom jakoby noise i batche tak bacha at si nespelete batch axis s time axis, jelikoz tady vlastne ani po batchich negenerujeme
        song = generator.predict(noise, verbose=1)

        song = song#.reshape(SONG_LENGTH,note_span_with_ligatures/2,2)
        matrixToMidi(song,'outputs/example_GAN_{}'.format(i))

rng = 0
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
