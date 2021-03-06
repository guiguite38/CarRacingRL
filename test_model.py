import tensorflow as tf
# from tensorflow.keras.optimizers import Adam
from train_model import process_state_image, ACTION_SPACE
import gym
import numpy as np


## Environnement
ENV = gym.envs.make("CarRacing-v0")

## Load Model
model = tf.keras.models.load_model('models\\trained_model')

if __name__ == "__main__":
    dones = False
    i = 0
    # Play once trained !
    # obs = process_state_image(ENV.reset()) # single frame version
    obs = np.array([process_state_image(ENV.reset()) for _ in range(4)]).reshape(96,96,12)
    # obs = np.array(obs).reshape(1,96,96,1)
    while dones == False and i <= 10000:
        print(f"[main.main] obs shape {obs.shape}")
        states = np.array([obs for _ in ACTION_SPACE])
        best_q_value = np.argmax(model.predict([states,ACTION_SPACE]))

        print(f"[main.main] model output {model.predict([states,ACTION_SPACE])}")
        
        action = ACTION_SPACE[best_q_value]
        print(f"[main.main] action {action}")        
        obs_unprocessed, rewards, dones, info = ENV.step(action)
        obs = np.concatenate((obs[:,:,3:],process_state_image(obs_unprocessed)), axis=2)
        ENV.render()
        i+=1
    print("there were " + str(i) + " steps")