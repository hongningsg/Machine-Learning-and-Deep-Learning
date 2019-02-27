import gym
import tensorflow as tf
import numpy as np
import random

# General Parameters
# -- DO NOT MODIFY --
ENV_NAME = 'CartPole-v0'
EPISODE = 200000  # Episode limitation
STEP = 200  # Step limitation in an episode
TEST = 10  # The number of tests to run every TEST_FREQUENCY episodes
TEST_FREQUENCY = 100  # Num episodes to run before visualizing test accuracy

# TODO: HyperParameters
GAMMA = 0.9 # discount factor
INITIAL_EPSILON = 0.6 # starting value of epsilon
FINAL_EPSILON = 0.1 # final value of epsilon
EPSILON_DECAY_STEPS = 100 # decay period

# Create environment
# -- DO NOT MODIFY --
env = gym.make(ENV_NAME)
epsilon = INITIAL_EPSILON
STATE_DIM = env.observation_space.shape[0]
ACTION_DIM = env.action_space.n

# Placeholders
# -- DO NOT MODIFY --
state_in = tf.placeholder("float", [None, STATE_DIM])
action_in = tf.placeholder("float", [None, ACTION_DIM])
target_in = tf.placeholder("float", [None])

# TODO: Define Network Graph
Replay_batch = []
Replay_size = 5000
Batch_size = 50
W_Layer_1 = tf.Variable(tf.random_normal([STATE_DIM, 100]))
bias_layer_1 = tf.Variable(tf.random_normal([100]))
logits = tf.matmul(state_in, W_Layer_1) + bias_layer_1
prediction = tf.nn.tanh(logits)
W_Layer_2 = tf.Variable(tf.zeros([100, ACTION_DIM]))
bias_layer_2 = tf.Variable(tf.zeros([ACTION_DIM]))

# TODO: Network outputs
q_values = tf.matmul(prediction, W_Layer_2) + bias_layer_2
q_action = tf.reduce_sum(tf.multiply(q_values, action_in), reduction_indices=1)

# TODO: Loss/Optimizer Definition
loss = tf.reduce_mean(tf.square(target_in - q_action))
optimizer = tf.train.AdamOptimizer().minimize(loss)

# Start session - Tensorflow housekeeping
session = tf.InteractiveSession()
session.run(tf.global_variables_initializer())


# -- DO NOT MODIFY ---
def explore(state, epsilon):
    """
    Exploration function: given a state and an epsilon value,
    and assuming the network has already been defined, decide which action to
    take using e-greedy exploration based on the current q-value estimates.
    """
    Q_estimates = q_values.eval(feed_dict={
        state_in: [state]
    })
    if random.random() <= epsilon:
        action = random.randint(0, ACTION_DIM - 1)
    else:
        action = np.argmax(Q_estimates)
    one_hot_action = np.zeros(ACTION_DIM)
    one_hot_action[action] = 1
    return one_hot_action


# Main learning loop
for episode in range(EPISODE):

    # initialize task
    state = env.reset()

    # Update epsilon once per episode
    epsilon -= epsilon / EPSILON_DECAY_STEPS

    # Move through env according to e-greedy policy
    for step in range(STEP):
        action = explore(state, epsilon)
        next_state, reward, done, _ = env.step(np.argmax(action))

        nextstate_q_values = q_values.eval(feed_dict={
            state_in: [next_state]
        })

        # TODO: Calculate the target q-value.
        # hint1: Bellman
        # hint2: consider if the episode has terminated

        Replay_batch.append((state, action, reward, next_state, done))
        if len(Replay_batch) > Replay_size:
            Replay_batch.pop(0)

        if (len(Replay_batch) > Batch_size):
            minibatch = random.sample(Replay_batch, Batch_size)
            state_batch = [data[0] for data in minibatch]
            action_batch = [data[1] for data in minibatch]
            reward_batch = [data[2] for data in minibatch]
            next_state_batch = [data[3] for data in minibatch]

            target_batch = []
            Q_value_batch = q_values.eval(feed_dict={
                state_in: next_state_batch
            })

            for i in range(Batch_size):
                sample_is_done = minibatch[i][4]
                if sample_is_done:
                    target_batch.append(reward_batch[i])
                else:
                    target = reward_batch[i] + GAMMA * np.max(Q_value_batch[i])
                    target_batch.append(target)
        # target = reward + GAMMA* np.max(nextstate_q_values)

            # Do one training step
            session.run([optimizer], feed_dict={
                target_in: target_batch,
                action_in: action_batch,
                state_in: state_batch
            })

        # Update
        state = next_state
        if done:
            break

    # Test and view sample runs - can disable render to save time
    # -- DO NOT MODIFY --
    if (episode % TEST_FREQUENCY == 0 and episode != 0):
        total_reward = 0
        for i in range(TEST):
            state = env.reset()
            for j in range(STEP):
                env.render()
                action = np.argmax(q_values.eval(feed_dict={
                    state_in: [state]
                }))
                state, reward, done, _ = env.step(action)
                total_reward += reward
                if done:
                    break
        ave_reward = total_reward / TEST
        print('episode:', episode, 'epsilon:', epsilon, 'Evaluation '
                                                        'Average Reward:', ave_reward)

env.close()
