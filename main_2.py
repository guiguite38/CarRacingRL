import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPooling2D

import cv2
import numpy as np
import gym
from collections import deque


env = gym.envs.make("CarRacing-v0") 

# Enjoy trained agent
obs = env.reset()
dones = False
i = 0

# input is 96*96*3 ints between 0 and 255
# output must be 12 values between -1 and 1 -6 TANH activation function

### PROF THING ###
# model = Sequential()
# model.add(Conv2D(filters=6, kernel_size=(7, 7), strides=3, activation='relu', input_shape=(96, 96, self.frame_stack_num)))
# model.add(MaxPooling2D(pool_size=(2, 2)))

# model.add(Conv2D(filters=12, kernel_size=(4, 4), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))

# model.add(Flatten())

# model.add(Dense(216, activation='relu'))
# model.add(Dense(len(self.action_space), activation=None))

# model.compile(loss='mean_squared_error', optimizer=Adam(lr=self.learning_rate, epsilon=1e-7))

# 8 convolution into 3 dense
model = Sequential()        model = Sequential()
        model.add(Conv2D(filters=6, kernel_size=(7, 7), strides=3, activation='relu', input_shape=(96, 96, self.frame_stack_num)))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(filters=12, kernel_size=(4, 4), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(216, activation='relu'))
        model.add(Dense(len(self.action_space), activation=None))
        model.compile(loss='me
model.add(Conv2D(filters=16, kernel_size=(8,8), padding='same', data_format="channels_last", input_shape=(96, 96, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(filters=16, kernel_size=(3,3), padding='same', data_format="channels_last", input_shape=(96, 96, 3)))
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

model.add(Dense(12, activation='relu'))


# discretisation
action_space    = [
    (-1, 1, 0.2), (0, 1, 0.2), (1, 1, 0.2), # Plein gaz + 20% frein
    (-1, 1,   0), (0, 1,   0), (1, 1,   0), # Plein gaz + 0% frein
    (-1, 0, 0.2), (0, 0, 0.2), (1, 0, 0.2), # 0% gaz + 20% frein
    (-1, 0,   0), (0, 0,   0), (1, 0,   0)  # 0% gaz + 0% frein
],

def process_state_image(state):
    state = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)
    state = state.astype(float)
    state /= 255.0
    return state

def generate_state_frame_stack_from_queue(deque):
    frame_stack = np.array(deque)
    # Move stack dimension to the channel dimension (stack, x, y) -> (x, y, stack)
    return np.transpose(frame_stack, (1, 2, 0))

# Training
def train_model(x_train,y_train,model):
    RENDER                        = True
    STARTING_EPISODE              = 1
    ENDING_EPISODE                = 1000
    SKIP_FRAMES                   = 2
    TRAINING_BATCH_SIZE           = 64
    SAVE_TRAINING_FREQUENCY       = 25
    UPDATE_TARGET_MODEL_FREQUENCY = 5
    
    for e in range(STARTING_EPISODE, ENDING_EPISODE+1):
        init_state = env.reset()
        init_state = process_state_image(init_state)

        total_reward = 0
        negative_reward_counter = 0
        state_frame_stack_queue = deque([init_state]*agent.frame_stack_num, maxlen=agent.frame_stack_num)
        time_frame_counter = 1
        done = False
        
        while True:
            if RENDER:
                env.render()

            current_state_frame_stack = generate_state_frame_stack_from_queue(state_frame_stack_queue)
            

            # train_state = []
            # train_target = []
            # state = current_state_frame_stack
            # # for state, action_index, reward, next_state, done in minibatch:
            # target = model.predict(np.expand_dims(state, axis=0))[0]
            # if done:
            #     target[action_index] = reward
            # else:
            #     t = self.target_model.predict(np.expand_dims(next_state, axis=0))[0]
            #     target[action_index] = reward + self.gamma * np.amax(t)
            # train_state.append(state)
            # train_target.append(target)
            # self.model.fit(np.array(train_state), np.array(train_target), epochs=1, verbose=0)
            # if self.epsilon > self.epsilon_min:
            #     self.epsilon *= self.epsilon_decay


            action = agent.act(current_state_frame_stack)

            reward = 0
            for _ in range(SKIP_FRAMES+1):
                next_state, r, done, info = env.step(action)
                reward += r
                if done:
                    break

            # If continually getting negative reward 10 times after the tolerance steps, terminate this episode
            negative_reward_counter = negative_reward_counter + 1 if time_frame_counter > 100 and reward < 0 else 0

            # Extra bonus for the model if it uses full gas # Do that kind of shenanigans later
            # if action[1] == 1 and action[2] == 0:
            #     reward *= 1.5

            total_reward += reward

            next_state = process_state_image(next_state)
            state_frame_stack_queue.append(next_state)
            next_state_frame_stack = generate_state_frame_stack_from_queue(state_frame_stack_queue)

            # agent.memorize(current_state_frame_stack, action, reward, next_state_frame_stack, done)

            if done or negative_reward_counter >= 25 or total_reward < 0:
                print('Episode: {}/{}, Scores(Time Frames): {}, Total Rewards(adjusted): {:.2}, Epsilon: {:.2}'.format(e, ENDING_EPISODE, time_frame_counter, float(total_reward), float(agent.epsilon)))
                break

# Play once trained ! 
while dones == False and i <= 10000:
    q_value = model.predict(obs)
    action = action_space[np.argmax(q_value)]
    obs, rewards, dones, info = env.step(action)
    env.render()
    i+=1
print("there were " + str(i) + " steps")