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

### Environnement ###
ENV = gym.envs.make("CarRacing-v0")

### Discretisation ###
# steer gas break in [-1,1] [0,1] [0,1] -> dtype float
ACTION_SPACE = np.array(
    [
        [-1, 1, 0.2],
        [0, 1, 0.2],
        [1, 1, 0.2],  # 3 directions (-1,0,1) + 100% gaz (1) + 20% frein (.2)
        [-1, 1, 0],
        [0, 1, 0],
        [1, 1, 0],  # Plein gaz + 0% frein
        [-1, 0, 0.2],
        [0, 0, 0.2],
        [1, 0, 0.2],  # 0% gaz + 20% frein
        [-1, 0, 0],
        [0, 0, 0],
        [1, 0, 0],  # 0% gaz + 0% frein
    ],
    dtype=float,
)

# input is 96*96*3 ints between 0 and 255
# output must be 12 values between -1 and 1 -6 TANH activation function


def network():
    """OUR NETWORK PREDICTING"""
    # define two sets of inputs
    input_state = Input(shape=(96, 96, 12))
    input_action = Input(shape=(3,))

    # the first branch operates on the first input = input_state
    x = Conv2D(filters=16, kernel_size=(8, 8), padding="same")(input_state)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(filters=16, kernel_size=(3, 3), padding="same")(x)
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
    """
    Output :
    - np_array(96,96,3)
    """
    # state = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)
    state = state.astype(float)
    state /= 255.0
    state = state.reshape(96, 96, 3)
    # print(f"[train_model.process_state_image] state type = {type(state)}")
    # print(f"[train_model.process_state_image] state shape = {state.shape}")
    return state


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
    a_list = []
    y = []
    for i in range(nb_episodes):
        print(f"--- episode {i} ---")
        s = np.array([process_state_image(game.reset()) for _ in range(4)]).reshape(96,96,12)
        cumul_rewards = 0
        e = False
        start_time = time.time()
        nb_frames = 0
        while e == False and time.time() - start_time < episode_time_limit:
            nb_frames += 1
            # chose action with epsilon greedy
            if np.random.random() < epsilon:  # epsilon
                a = random.choice(ACTION_SPACE)
                q_value_predicts = []
                idx = 0
                q_value_predicts.append(model.predict([s.reshape(1,96,96,12), a.reshape(1,3)])[0])
            else:
                states = np.array([s for _ in ACTION_SPACE])
                q_value_predicts = model.predict([states, ACTION_SPACE])
                idx = np.argmax(q_value_predicts)
                a = ACTION_SPACE[idx]

            s1_unprocessed, r, e, _ = game.step(a)

            if a[1] == 1 and a[2] == 0:
                r *= 1.5
            s1 = np.concatenate((s[:,:,3:],process_state_image(s1_unprocessed)), axis=2)

            # cumul_rewards += r
            # save expected q_value ce sont les y
            # save chosen action ce sont les x
            s_list.append(s)
            a_list.append(a)

            q_value_predict = q_value_predicts[idx]
            new_states = np.array([s1 for _ in ACTION_SPACE])
            q_value_future = np.max([model.predict([new_states, ACTION_SPACE])])

            # Q[s,a] = Q[s,a] + alpha* (r + gamma * np.max(Q[s1,:]) - Q[s,a])
            q_value = q_value_predict + alpha * (
                r + gamma * q_value_future - q_value_predict
            )

            print(f"[main.generate_x_y] q_value {q_value}\treward {r}")
            y.append(q_value[0])
            s = s1

            if render:
                ENV.render()
        print(f"[main.generate_x_y] frames computed : {nb_frames}")
    return s_list, a_list, y


if __name__ == "__main__":
    alpha = 0.1
    gamma = 0.9
    epsilon = 0.5
    episode_time_limit=20
    model = network()
    model.compile(
        loss="mean_squared_error", optimizer=Adam(learning_rate=1e-3), metrics=["mse"]
    )

    training_cycles = 10

    for cycle in range(training_cycles):
        print(f"Entering cycle {cycle}")
        epsilon = epsilon * 0.85
        episode_time_limit = min(episode_time_limit *1.1,50)
        s, a, y = generate_x_y(
            model,
            ENV,
            epsilon=epsilon,
            nb_episodes=3,
            episode_time_limit = episode_time_limit,
            render=True,
        )
        s = np.array(s)
        a = np.array(a)
        y = np.array(y)
        model.fit([s, a], y, verbose=2)

    print(f"[main.__main__] len y {len(y)}")
    print(f"[main.__main__] len states {len(s)}")
    print(f"[main.__main__] len actions {len(a)}")
    model.save("models\\trained_model")
