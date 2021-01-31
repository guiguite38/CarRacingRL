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
    obs = np.array(obs).reshape(1,96,96,1)
    while dones == False and i <= 10000:
        #print(f"[main.main] observation format {obs}")
        print(f"[main.main] observation length {len(obs)}")
        #q_value = model.predict([obs,ACTION_SPACE])
        # q_values = [model.predict([[obs]*12,ACTION_SPACE])

        q_values = [model.predict([[obs],[a]]) for a in ACTION_SPACE]
        
        action = ACTION_SPACE[np.argmax(q_values)]
        print(f"[main.main] action {action}")
        obs, rewards, dones, info = ENV.step(action)
        ENV.render()
        i+=1
    print("there were " + str(i) + " steps")