
from collections import deque, namedtuple

import DenseLayer as dl
import numpy as np
import gym
import PIL.Image
import random
import utils
import time

from pyvirtualdisplay import Display

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.losses import MSE
from tensorflow.keras.optimizers import Adam

# Set the random seed for TensorFlow
tf.random.set_seed(utils.SEED)
random.seed(utils.SEED)

NUM_STEPS_FOR_UPDATE = 4
MEMORY_SIZE = int(1e5)
GAMMA = 0.995
ALPHA = 1e-3

Display(visible=0, size=(840, 480)).start()

env = gym.make('LunarLander-v2', render_mode='rgb_array')
env.reset()
PIL.Image.fromarray(env.render())
# Reset the environment and get the initial state.
current_state = env.reset()

state_size = env.observation_space.shape
num_actions = env.action_space.n

print(f'State Shape: {state_size}')
print(f'Number of actions: {num_actions}')

INNER_UNITS_1 = 64
INNER_UNITS_2 = INNER_UNITS_1

targetQ_NNf = dl.NeuralNetwork_flow([
                dl.InputLayer(units=state_size[0]),
                dl.DenseLayer(units=INNER_UNITS_1, type="relu"),
                dl.DenseLayer(units=INNER_UNITS_2, type="relu"),
                dl.DenseLayer(units=num_actions, type='linear')])

Q_NNf = dl.NeuralNetwork_flow([
                dl.InputLayer(units=state_size[0]),
                dl.DenseLayer(units=INNER_UNITS_1, type="relu", optimizer=dl.AdamOptimizer(0.001)),
                dl.DenseLayer(units=INNER_UNITS_2, type="relu", optimizer=dl.AdamOptimizer(0.001)),
                dl.DenseLayer(units=num_actions, type='linear', optimizer=dl.AdamOptimizer(0.001))])

Q_network = Sequential([
    Input(shape=state_size),
    Dense(units=INNER_UNITS_1, activation='relu'),
    Dense(units=INNER_UNITS_2, activation='relu'),
    Dense(units=num_actions, activation='linear')])

targetQ_network = Sequential([
    Input(shape=state_size),
    Dense(units=INNER_UNITS_1, activation='relu'),
    Dense(units=INNER_UNITS_2, activation='relu'),
    Dense(units=num_actions, activation='linear')])

optimizer = Adam(learning_rate=1e-3)

# Store experiences as named tuples
experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

def agent_learn(experiences, gamma):
    # Calculate the loss
    with tf.GradientTape() as tape:
        loss = compute_loss(experiences, gamma, Q_network, targetQ_network)
    
    # tf.print(loss)

    # Get the gradients of the loss with respect to the weights.
    gradients = tape.gradient(loss, Q_network.trainable_variables)

    # for grad in gradients:
    #     for i in range(grad.shape[0]):
    #         tf.print(grad[i])

    # Update the weights of the q_network.
    optimizer.apply_gradients(zip(gradients, Q_network.trainable_variables))

    # tf.print(compute_loss(experiences, gamma, Q_network, targetQ_network))

    # update the weights of target q_network
    utils.update_target_network(Q_network, targetQ_network)

def compute_loss(experiences, gamma, q_network, target_q_network):
    # Unpack the mini-batch of experience tuples
    states, actions, rewards, next_states, done_vals = experiences
    # Compute max Q^(s,a)
    predict_state = target_q_network(next_states)
    max_qsa = tf.reduce_max(predict_state, axis=-1)

    # Set y = R if episode terminates, otherwise set y = R + γ max Q^(s,a).
    y_targets = rewards + gamma * max_qsa * (1 - done_vals)

    # Get the q_values and reshape to match y_targets
    q_values = q_network(states)
    q_values = tf.gather_nd(q_values, tf.stack([tf.range(q_values.shape[0]),
                                                tf.cast(actions, tf.int32)], axis=1))
    
    # Compute the loss: y, output => 2*(y_targets - q_values) => q_values = output[actions]
    return MSE(y_targets, q_values)

def get_action(q_values, epsilon=0.0):
    if random.random() > epsilon:
        return np.argmax(q_values)
    else:
        return random.choice(np.arange(4))

def compute_train_step(experiences, gamma, 
                       Q_network:dl.NeuralNetwork_flow, 
                       targetQ_network:dl.NeuralNetwork_flow, isPrint:bool=False):
    # Unpack the mini-batch of experience tuples
    states, actions, rewards, next_states, done_vals = experiences
    states = states.numpy().T
    actions = actions.numpy().astype(np.uint8)
    rewards = rewards.numpy()
    next_states = next_states.numpy().T
    done_vals = done_vals.numpy()
    # Compute max Q^(s,a)
    predict_state = targetQ_network.predict(next_states)
    max_qsa = np.max(predict_state, axis=0)
    # Set y = R if episode terminates, otherwise set y = R + γ max Q^(s,a).
    y_targets = rewards + gamma * max_qsa * (1 - done_vals)
    if isPrint == False:
        # Step of Q-network training
        tmp_Q_predicts = Q_network.predict(states)
        transferMtx = np.zeros(tmp_Q_predicts.shape, dtype="float32")
        for i in range(len(actions)):
            transferMtx[actions[i],i] = 1
        Q_output = np.array([tmp_Q_predicts[actions[i],i] for i in range(len(actions))])
        # print(np.sum((y_targets - Q_output)**2)/len(y_targets))
        Q_network.backprop((2/len(y_targets))*(y_targets-Q_output)*transferMtx)
        qLayers = Q_network.getLayers()
        trgtQLayers = targetQ_network.getLayers()
        for i in range(1, len(qLayers)):
            trgtQLayers[i].softUpdate(qLayers[i].getWeights(), qLayers[i].getBias())
    else:
        tmp_Q_predicts = Q_network.predict(states)
        Q_output = np.array([tmp_Q_predicts[actions[i],i] for i in range(len(actions))])
        print(np.sum((y_targets - Q_output)**2)/len(y_targets))

start = time.time()

num_episodes = 2000
max_num_timesteps = 1000

total_point_history = []

num_p_av = 100    # number of total points to use for averaging
epsilon = 1.0     # initial ε value for ε-greedy policy

# Create a memory buffer D with capacity N
memory_buffer = deque(maxlen=MEMORY_SIZE)

# Set the target network weights equal to the Q-Network weights
targetQ_network.set_weights(Q_network.get_weights())
tf_Qweights = Q_network.get_weights()

qNN_lrs = Q_NNf.getLayers()
trgtQNN_lrs = targetQ_NNf.getLayers()
for i in range(1, len(qNN_lrs)):
        tf_Qw_lr = tf_Qweights[(i-1)*2]
        tf_Qb_lr = tf_Qweights[(i-1)*2+1].reshape((-1,1))
        # qNN_lrs[i].update(tf_Qw_lr.T, tf_Qb_lr)
        # trgtQNN_lrs[i].update(tf_Qw_lr.T, tf_Qb_lr)
        trgtQNN_lrs[i].update(qNN_lrs[i].getWeights(), qNN_lrs[i].getBias())

for i in range(num_episodes):

    # Reset the environment to the initial state and get the initial state
    state = env.reset()
    state = state[0]
    total_points = 0

    for t in range(max_num_timesteps):
        # From the current state S choose an action A using an ε-greedy policy
        state_qn = np.expand_dims(state, axis=0)  # state needs to be the right shape for the q_network
        q_values = Q_network(state_qn)
        action = utils.get_action(q_values, epsilon)
        # q_values = Q_NNf.predict(state_qn.T)
        # action = get_action(q_values, epsilon)

        # Take action A and receive reward R and the next state S'
        next_state, reward, done, _, _ = env.step(action)

        # Store experience tuple (S,A,R,S') in the memory buffer.
        # We store the done variable as well for convenience.
        memory_buffer.append(experience(state, action, reward, next_state, done))

        # Only update the network every NUM_STEPS_FOR_UPDATE time steps.
        update = utils.check_update_conditions(t, NUM_STEPS_FOR_UPDATE, memory_buffer)

        if update:
            # Sample random mini-batch of experience tuples (S,A,R,S') from D
            experiences = utils.get_experiences(memory_buffer)
            # experiences = random.sample(memory_buffer, k=65)
            
            # Set the y targets, perform a gradient descent step,
            # and update the network weights.
            agent_learn(experiences, GAMMA)
            # compute_train_step(experiences, GAMMA, Q_NNf, targetQ_NNf)
            # lossValue = compute_loss(experiences,GAMMA,Q_network,targetQ_network)
        
        state = next_state.copy()
        total_points += reward
        if done:
            break
    
    # print(f"{i} iteration: {compute_loss(experiences,GAMMA,Q_network,targetQ_network)}")
    # compute_train_step(experiences, GAMMA, Q_NNf, targetQ_NNf,isPrint=True)

    total_point_history.append(total_points)
    av_latest_points = np.mean(total_point_history[-num_p_av:])
    
    # Update the ε value
    epsilon = utils.get_new_eps(epsilon)

    print(f"\rEpisode {i+1} | Total point average of the last {num_p_av} episodes: {av_latest_points:.2f}", end="")

    if (i+1) % num_p_av == 0:
        print(f"\rEpisode {i+1} | Total point average of the last {num_p_av} episodes: {av_latest_points:.2f}")

    # We will consider that the environment is solved if we get an
    # average of 200 points in the last 100 episodes.
    if av_latest_points >= 200.0:
        print(f"\n\nEnvironment solved in {i+1} episodes!")
        Q_network.save('lunar_lander_model.h5')
        break

tot_time = time.time() - start

print(f"\nTotal Runtime: {tot_time:.2f} s ({(tot_time/60):.2f} min)")