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
import time

### Environnement ###
ENV = gym.envs.make("CarRacing-v0")

### Discretisation ###
ACTION_SPACE = (
    (-1, 1, 0.2), (0, 1, 0.2), (1, 1, 0.2), # 3 directions (-1,0,1) + 100% gaz (1) + 20% frein (.2) 
    (-1, 1,   0), (0, 1,   0), (1, 1,   0), # Plein gaz + 0% frein
    (-1, 0, 0.2), (0, 0, 0.2), (1, 0, 0.2), # 0% gaz + 20% frein
    (-1, 0,   0), (0, 0,   0), (1, 0,   0)  # 0% gaz + 0% frein
)

# input is 96*96*3 ints between 0 and 255
# output must be 12 values between -1 and 1 -6 TANH activation function


def network():
    """OUR NETWORK PREDICTING"""
    # define two sets of inputs
    input_state = Input(shape=(96,96,1))
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
    z = Dense(1, activation="relu")(z)

    # our model will accept the inputs of the two branches and
    # then output a single value
    model = Model(inputs=[input_state, input_action], outputs=z)
    return model


def process_state_image(state):
    state = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)
    state = state.astype(float)
    state /= 255.0
    return state
    

def generate_x_y(model, game, nb_episodes=1000, epsilon=0.3):
    print(f"[main.generate_x_y] entering generate_x_y, launching {nb_episodes} episodes")
    s_list = []
    a_list = []
    y = []
    for i in range(nb_episodes):
        print(f"--- episode {i} ---")
        s=process_state_image(game.reset())
        s=np.array(s).reshape(96,96,1)
        cumul_rewards = 0
        e = False
        start_time = time.time()
        nb_frames = 0
        while e==False and time.time()-start_time <5:
            print(f"[main.generate_x_y] entering while loop - time = {time.time()-start_time}")
            nb_frames+=1
            # chose action with epsilon greedy
            if np.random.random() < 0: #should be epsilon
                a = random.choice(ACTION_SPACE)
            else:
                # ((96, 96, 3), (3, )) TODO : make this our input, can work with concat

                # TODO CHECK INPUT DATA FOR MODEL PREDS
                                
                idx = np.argmax([model.predict([[s],[a]]) for a in ACTION_SPACE])
                a = ACTION_SPACE[idx]

            s1_unprocessed, r, e, _ = game.step(a)
            s1 = process_state_image(s1_unprocessed)
            s1=np.array(s1).reshape(96,96,1)

            # cumul_rewards += r
            # save expected q_value ce sont les y
            # save chosen action ce sont les x
            # x.append([s,a])
            s_list.append(s)
            a_list.append(a)
            
            # Q[s,a] = Q[s,a] + alpha* (r + gamma * np.max(Q[s1,:]) - Q[s,a])
            q_value = model.predict([[s],[a]]) + alpha * (r + gamma * np.max([model.predict([[s1],[a]]) for a in ACTION_SPACE]) - model.predict([[s],[a]]))
            
            # print(f"[main.generate_x_y] q_value {q_value[0][0]}")
            y.append(q_value[0][0])
            s = s1
        print(f"[main.generate_x_y] frames computed : {nb_frames}")
    return s_list, a_list, y


if __name__ == '__main__':
    alpha = 0.1
    gamma = 0.9

    model = network()
    model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=1e-3), metrics=['accuracy', 'mse'])

    training_cycles = 3

    for cycle in range(training_cycles):
        print(f"Entering cycle {cycle}")
        s, a, y = generate_x_y(model,ENV, nb_episodes=2)
        # print(f"[main.__main__] x shape : {np.array(x).shape} expected [?, (96,96,1),(3,1,1)]")
        # print(f"[main.__main__] y shape : {np.array(y).shape} expected (?, 1)")
        # print(f"[main.__main__] x0 shape : {np.array(x[0]).shape} expected [?, (96,96,1),(3,1,1)]")
        # print(f"[main.__main__] x0 : {np.array(x[0])}")

        # s = [np.array(x[i][0]).reshape(96,96,1) for i in range(len(x))]
        # a = [np.array(x[i][1]).reshape(3) for i in range(len(x))]
        
        # print(f"[main.__main__] x processed : {np.array(x_processed).shape}")
        # print(f"[main.__main__] x0 processed : {np.array(x_processed[0]).shape}")
        # print(f"[main.__main__] x00 : {np.array(x_processed[0][0][0]).shape}")
        # print(f"[main.__main__] x00 : {np.array(x_processed[0][0][1]).shape}")
        
        print(f"[main.__main__] y {y} of len {len(y)}")
        print(f"[main.__main__] s {len(s)}")
        print(f"[main.__main__] a {a}")

        model.fit([s,a],np.array(y),verbose=2)
        # model.fit(x,y,verbose=2)
    model.save('models\\trained_model')
    