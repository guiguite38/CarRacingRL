import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Concatenate
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Input, Model

import cv2
from cv2 import cvtColor
import numpy as np
import gym
from collections import deque
import random

### Environnement ###
ENV = gym.envs.make("CarRacing-v0")

### Discretisation ###
ACTION_SPACE = np.array((
    (-1, 1, 0.2), (0, 1, 0.2), (1, 1, 0.2), # 3 directions (-1,0,1) + 100% gaz (1) + 20% frein (.2) 
    (-1, 1,   0), (0, 1,   0), (1, 1,   0), # Plein gaz + 0% frein
    (-1, 0, 0.2), (0, 0, 0.2), (1, 0, 0.2), # 0% gaz + 20% frein
    (-1, 0,   0), (0, 0,   0), (1, 0,   0)  # 0% gaz + 0% frein
))

# input is 96*96*3 ints between 0 and 255
# output must be 12 values between -1 and 1 -6 TANH activation function

def process_state_image(state):
    state = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)
    state = state.astype(float)
    state /= 255.0
    return state


def network():
    """OUR NETWORK PREDICTING"""
    # define two sets of inputs
    input_state = Input(shape=(96,96,3))
    input_action = Input(shape=(3,))

    # the first branch operates on the first input = input_state
    x = Conv2D(filters=16, kernel_size=(8,8), padding='same')(input_state)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(filters=16, kernel_size=(3,3), padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = BatchNormalization()(x)
    x = Flatten()(x)
    x = Model(inputs=input_state, outputs=x)

    # combine the output of the two branches
    combined = Concatenate(axis=1)([x.output, input_action])

    # apply a FC layer and then a regression prediction on the
    # combined outputs
    z = Dense(256, activation="relu")(combined)
    # z = Dropout(0.5)(z)
    z = Dense(128, activation="relu")(z)
    z = Dense(12, activation="sigmoid")(z)

    # our model will accept the inputs of the two branches and
    # then output a single value
    model = Model(inputs=[input_state, input_action], outputs=z)
    return model


def network2():
    '''OUR NETWORK'''
    # ((96, 96, 3), (3, )) TODO : make this our input, can work with concat
    model = Sequential()
    model.add(Conv2D(filters=16, kernel_size=(8,8), padding='same', data_format="channels_last", input_shape=(96, 96, 3))) # !! TODO : Change this to be [s,a]
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(filters=16, kernel_size=(3,3), padding='same', data_format="channels_last"))
    model.add(BatchNormalization()) 
    # model.add(Activation('relu'))

    model.add(Flatten())

    model.add(Dense(160,activation='relu'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(160))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(len(ACTION_SPACE), activation='softmax'))

    return model


def generate_x_y(model, game, nb_episodes=1000, epsilon=0.3):
    for i in range(nb_episodes):
        s=game.reset()
        x = []
        y = []
        cumul_rewards = 0
        e = False
        
        while True:
            # chose action with epsilon greedy
            if np.random.random() < 0: #should be epsilon
                a = random.choice(ACTION_SPACE)
            else:
                # ((96, 96, 3), (3, )) TODO : make this our input, can work with concat

                # print(f"[main.generate_x_y] state {s}")
                # print(f"[main.generate_x_y] actions {actions)}")
                print(f"[main.generate_x_y] prediction {model.predict([s,ACTION_SPACE[0]])}")
                # TODO CHECK INPUT DATA FOR MODEL PREDS
                
                idx = np.argmax([model.predict([s,a]) for a in ACTION_SPACE])
                a = ACTION_SPACE[idx]
            s1, r, e, _ = game.move(a)
            # cumul_rewards += r
            
            # ================================= #
            # Use DQN to estimate q_value below #
            # ================================= #
        
            # save expected q_value ce sont les y
            # save chosen action ce sont les x
            x.append([s,a])

            # Q[s,a] = Q[s,a] + alpha* (r + gamma * np.max(Q[s1,:]) - Q[s,a])
            q_value = model.predict([s,a]) + alpha * (r + gamma * np.max(model.predict[s1,:]) - model.predict[s,a])
            y.append(q_value)
            s = s1
    
            if e == True:
                break
    return x,y


if __name__ == '__main__':
    model = network()
    model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=1e-3), metrics=['accuracy', 'mse'])

    training_cycles = 3
    for i in range(training_cycles):
        x,y = generate_x_y(model,ENV)
        model.fit(x,y)
    
    dones = False
    i = 0
    # Play once trained !
    obs = ENV.reset() 
    while dones == False and i <= 10000:
        print(f"[main_2.main] observation format {obs}")
        print(f"[main_2.main] observation length {len(obs)}")        
        q_value = model.predict(obs)
        action = ACTION_SPACE[np.argmax(q_value)]
        obs, rewards, dones, info = ENV.step(action)
        ENV.render()
        i+=1
    print("there were " + str(i) + " steps")