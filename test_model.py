import tensorflow as tf
# from tensorflow.keras.optimizers import Adam
from train_model import process_state_image, ACTION_SPACE
import gym
import numpy as np


## Environnement
ENV = gym.envs.make("CarRacing-v0")

## Load Model
model = tf.keras.models.load_model('models\\trained_model')
# model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=1e-3), metrics=['accuracy', 'mse'])

if __name__ == "__main__":
    dones = False
    i = 0
    # Play once trained !
    obs = process_state_image(ENV.reset())
    # obs = np.array(obs).reshape(1,96,96,1)
    while dones == False and i <= 10000:
        print(f"[main.main] obs shape {obs.shape}")
        #q_value = model.predict([obs,ACTION_SPACE])
        # q_values = [model.predict([[obs]*12,ACTION_SPACE])
        states = np.array([obs for _ in ACTION_SPACE])
        best_q_value = np.argmax(model.predict([states,ACTION_SPACE]))
        
        action = ACTION_SPACE[best_q_value]
        print(f"[main.main] action {action}")        
        obs, rewards, dones, info = ENV.step(action)     
        obs = process_state_image(obs)
        ENV.render()
        i+=1
    print("there were " + str(i) + " steps")