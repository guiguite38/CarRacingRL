import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Concatenate
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Input, Model

# import cv2
# from cv2 import cvtColor
import numpy as np
import gym
from collections import deque
import random
import time
from train_model import process_state_image, ACTION_SPACE

### Environnement ###
ENV = gym.envs.make("CarRacing-v0")

# input is 96*96*3 ints between 0 and 255
# output must be 12 values between -1 and 1 -6 TANH activation function


def network():
    """OUR NETWORK PREDICTING"""
    # define two sets of inputs
    input_state = Input(shape=(96, 96, 12))

    # the first branch operates on the first input = input_state
    x = Conv2D(filters=16, kernel_size=(8, 8), padding="same")(input_state)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(filters=16, kernel_size=(3, 3), padding="same")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = BatchNormalization()(x)
    x = Flatten()(x)

    # apply a FC layer and then a regression prediction on the
    z = Dense(256, activation="relu")(x)
    # z = Dropout(0.5)(z)
    z = Dense(128, activation="relu")(z)
    z = Dense(12)(z)

    # our model will accept the inputs of the two branches and
    # then output a single value
    model = Model(inputs=input_state, outputs=z)
    return model


def generate_x_y(
    model,
    game,
    nb_episodes=10,
    epsilon=0.3,
    episode_time_limit=40,
    render=False,
):
    print(
        f"[main.generate_x_y] entering generate_x_y, launching {nb_episodes} episodes"
    )
    s_list = []
    y = []
    for i in range(nb_episodes):
        print(f"--- episode {i} ---")
        original_state = process_state_image(game.reset())
        s = np.array([original_state for _ in range(4)]).reshape(96,96,12)
        cumul_rewards = 0
        e = False
        start_time = time.time()
        nb_frames = 0
        nb_negative_rewards = 0
        # while e == False and time.time() - start_time < episode_time_limit and nb_negative_rewards < 10:
        while e == False and nb_negative_rewards < 10:
            nb_frames += 1
            # chose action with epsilon greedy
            if np.random.random() < epsilon:  # epsilon
                # if np.random.random() < 0.5:
                #     # epsilon booster
                #     a = ACTION_SPACE[0]
                #     q_value_predicts = []
                #     idx = 0
                #     q_value_predicts.append(model.predict([s.reshape(1,96,96,12), a.reshape(1,3)])[0])
                # else:
                q_value_predicts = model.predict(s.reshape(1,96,96,12))[0]
                idx = np.random.choice(len(ACTION_SPACE))
                a = ACTION_SPACE[idx]
            else:
                q_value_predicts = model.predict(s.reshape(1,96,96,12))[0]
                idx = np.argmax(q_value_predicts)
                a = ACTION_SPACE[idx]

            s1_unprocessed, r, e, _ = game.step(a)

            # We check for negative rewards. If too many occur in a row, we terminate the episode
            nb_negative_rewards+= 1 if nb_frames > 30 and r < 0 else 0

            if a[1] == 1 and a[2] == 0:
                r *= 1.5
            cumul_rewards += r
            s1 = np.concatenate((s[:,:,3:],process_state_image(s1_unprocessed)), axis=2)

            # save expected q_value ce sont les y
            # save chosen action ce sont les x
            s_list.append(s)

            q_value_predict = q_value_predicts[idx]
            q_value_future = np.max([model.predict(s1.reshape(1,96,96,12))])

            q_value = q_value_predict + alpha * (r + gamma * q_value_future - q_value_predict)
            # q_value = r + gamma * q_value_future
            
            # print(f"[main.generate_x_y] q_value {q_value}   \taction {a} \treward {r}")

            #   In this version only the explored state is modified :
            q_value_predicts[idx] = q_value
            y.append(q_value_predicts)
            s = s1

            if render:
                ENV.render()
        print(f"[main.generate_x_y] frames computed : {nb_frames}")
        print(f"[main.generate_x_y] cumul reward : {cumul_rewards}")
    return s_list, y


if __name__ == '__main__':
    alpha = 0.5
    gamma = 0.9
    epsilon = 0.5
    batch_size = 256
    nb_episodes = 2 # peut être augmenter le nombre de données pour avoir une meilleur loss ? là ça semble nul, faudrait observer avec tensorboard
    episode_time_limit=20 # faire fortement grandir la limite épisode peut être pertinent avec le boost
    model = network()
    model.compile(
        loss="logcosh", optimizer=Adam(learning_rate=1e-3), metrics=["mse","mae"] # tester avec logcosh pour gérer les outliers mal prédits
    )

    training_cycles = 600

    s_buffer = np.array([])
    y_buffer = np.array([])
    for cycle in range(training_cycles):
        print(f"Entering cycle {cycle}")
        epsilon = epsilon * 0.85
        # episode_time_limit = min(episode_time_limit *1.1,60) 
        s, y = generate_x_y(
            model,
            ENV,
            epsilon=epsilon,
            nb_episodes=nb_episodes,
            episode_time_limit = episode_time_limit,
            render=False,
        )
        if len(s_buffer) == 0:
            s_buffer = s
            y_buffer = y
        else:
            np.append(s_buffer,s)
            np.append(y_buffer,y) # to be normalized ?
        # ##
        # norm = np.linalg.norm(y)
        # y_norm = y/norm
        # ##
        bestQ = [np.max(np.abs(y) ) for y in y_buffer]
        batch_idx = np.random.choice(np.arange(len(s_buffer)), min(batch_size, len(s_buffer)), replace=False, p = bestQ/np.sum(bestQ))
        model.fit(np.array(s_buffer)[batch_idx],np.array(y_buffer)[batch_idx],verbose=2, validation_split=0.1)
        if cycle % 50 ==0:
            model.save(f"models\\trained_model_{cycle}_v5")
            # purge 2/10th of memory
            mem_idx = np.random.choice(np.arange(len(s_buffer)), len(s_buffer)*80//100, replace=False)#, p = bestQ/np.sum(bestQ))
            s_buffer = np.array(s_buffer)[mem_idx]
            y_buffer = np.array(y_buffer)[mem_idx]

    model.save("models\\trained_model")
    print(f"[main.__main__] len y {len(y)}")
    print(f"[main.__main__] len states {len(s)}")
    # print(f"[main.__main__] len actions {len(a)}")

