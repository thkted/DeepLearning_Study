import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import random
import dqn
from collections import deque
import gym

env = gym.make('CartPole-v0')

input_size = env.observation_space.shape[0]
output_size = env.action_space.n

dis = 0.9
REPLAY_MEMORY = 50000

def replay_train(mainDQN, targetDQN, train_batch):
    x_stack = np.empty(0).reshape(0, input_size)
    y_stack = np.empty(0).reshape(0, output_size)

    for state, action, reward, next_state, done in train_batch:
        Q = mainDQN.predict(state)
        
        if done:
            Q[0, action] = -100
        else:
            Q[0, action] = reward + dis * np.max(targetDQN.predict(next_state))
        
        x_stack = np.vstack([x_stack, state])
        y_stack = np.vstack([y_stack, Q])
    
    return mainDQN.update(x_stack, y_stack)

def bot_play(mainDQN):
    s = env.reset()
    reward_sum = 0

    while True:
        env.render()
        a = np.argmax(mainDQN.predict(s))
        s, reward, done, _ = env.step(a)
        reward_sum += reward

        if done:
            print('Total score: {}'.format(reward_sum))
            break

def get_copy_var_ops(*, dest_scope_name='target', src_scope_name='main'):
    op_holder = []

    src_vars = tf.GraphKeys.TRAINABLE_VARIABLES, scope=src_scope_name
    dest_vars = tf.GraphKeys.TRAINABLE_VARIABLES, scope=dest_scope_name

    for src_var, dest_var in zip(src_vars, dest_vars):
        op_holder.append(dest_vars.assign(src_vars.value()))
    
    return op_holder

def main():
    max_episodes = 5000
    replay_buffer = deque()

    with tf.Session() as sess:
        mainDQN = dqn.DQN(sess, input_size, output_size, name='main')
        targetDQN = dqn.DQN(sess, input_size, output_size, name='main')
        tf.global_variables_initializer().run()

        copy_ops = get_copy_var_ops(dest_scope_name='target', 
        src_scope_name='main')

        sess.run(copy_ops)

        for episode in range(max_episodes):
            e = 1. / ((episode / 10) + 1)
            done = False
            step_count = 0

            state = env.reset()

            while not done:
                if np.random.rand(1) < e:
                    action = env.action_space.sample()
                else:
                    action = np.argmax(mainDQN.predict(state))
                
                next_state, reward, done, _ = env.step(action)

                if done:
                    reward = -100

                replay_buffer.append((state, action, reward, next_state, done))

                if len(replay_buffer) > REPLAY_MEMORY:
                    replay_buffer.popleft()
                
                state = next_state
                step_count += 1
                if step_count > 10000:
                    break
            
            print('Episode: {} / steps: {}'.format(episode, step_count))

            if step_count > 10000:
                pass
                break
            
            if episode % 10 == 1:
                for _ in range(50):
                    minibatch = random.sample(replay_buffer, 10)
                    loss, _ = replay_train(mainDQN, targetDQN, minibatch)
                
                print('Loss: ', loss)
                sess.run(copy_ops)
        
        bot_play(mainDQN)

if __name__ == '__main__':
    main()
