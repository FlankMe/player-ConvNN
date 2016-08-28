# -*- coding: utf-8 -*-
"""
Implementation of a Convolutional Neural Network that applies the deep
Q-learning algorithm to play PyGame video games.

As inputs, it accepts:
    - image: array, image made of raw pixels captured from the game screen;
    - reward: float, reward received at that particular state;
    - is_terminal: bool, indicates if the player is at a terminal state;
and returns:
    - an integer as chosen action;

The CNN is then applied on the game Asteroids, a simple arcade game from my 
GitHub profile. 

Created on Sat Jun 25 18:19:59 2016
@author: Riccardo Rossi
"""

import time
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
np.random.seed(int(time.time()))
import matplotlib.pyplot as plt
import matplotlib as plt2

# Hyperparameters
FORCED_TRAINING = False  # Set to False unless the Agent is meant to be trained

STATE_FRAMES = 3         # number of states/images used for taking a decision
DISCOUNTING_RATE = 0.9  # discount of future rewards
NUMBER_ALLOWED_ACTIONS = 3     # number of possible actions that can be chosen

#####   CNN's architecture   ##################################################
# Make sure the architecture is consistent if you choose to change it
# In order to preserve flexibility, all layers are appended to lists
#
# (CONV -> MAXPOOL -> DROPOUT)*n -> (FEED_FWD)*m -> STATE_VALUE_ACTION_ESTIMATE
#
###############################################################################
RESIZED_SCREEN = 84      # size of the reduced screen fed to the CNN
CONVOLUTION_FILTER_VECTOR = [6, 6, 4]
CONVOLUTION_STRIDE_VECTOR = [3, 3, 2]
CONVOLUTION_KERNEL_VECTOR = [16, 16, 36]
MAXPOOL_VECTOR = [1, 1, 1]
CONVOLUTION_INPUT_VECTOR = [STATE_FRAMES] + CONVOLUTION_KERNEL_VECTOR[:-1]
FEED_FWD_VECTOR = [(3**2) * CONVOLUTION_KERNEL_VECTOR[-1], 64, 
                   NUMBER_ALLOWED_ACTIONS]  
DROPOUT_PROBABILITY = 1.

# Exploration/exploitations parameters
EPSILON_EXPLORATION_AT_REGIME = 0.1      # Prob of exporation during training
PROB_KEEP_IN_MEMORY_IF_UNEVENTFUL = 1. # Prob of storing an observation
MAX_TIME = 150000
MIN_MEMORY_FOR_TRAINING = 1000
TIME_EXPLORATION_AT_REGIME = 100000

# Initialization parameters
INITIALIZATION_STDDEV = 0.1
INITIALIZATION_MEAN = 0.01
INITIALIZATION_BIAS = -0.001

# AI sees the screen in grey scale, and it is sensitive to 1 colour only
COLOUR = 2
LAST_STATE_IDX, ACTION_IDX, REWARD_IDX, CURR_STATE_IDX, TERMINAL_IDX = range(5)
PARAMETERS_FILE_PATH = 'Parameters_CNN.ckpt'

# Training parameters
ALPHA = 1e-3               # learning rate
MINI_BATCH_SIZE = 32       # Number of observations to train on at every step
TRAINING_STEPS = 10        # Stages to observe between training epochs
REPLAY_MEMORY = 50000      # Total number of previous transitions to remember
MAX_OBSERVATIONS_IN_FILE = 10000   # Number of observations to be saved on file



class Agent():


    def __init__(self, field_of_view=504, memory_file_path=[]):
        '''At initialization, the class requires the size of the screen 
            it will receive as input. Note that the input is then expected 
            to be of shape = [1, field_of_view, field_of_view, 3]
            '''
        
        # Initialization of useful variables and constants
        self._FIELD_OF_VIEW = field_of_view       
        self._last_state = []
        self._last_action = np.zeros([NUMBER_ALLOWED_ACTIONS])
        self._previous_observations = []
        self._PARAMETERS_FILE_PATH = PARAMETERS_FILE_PATH 
        self._MEMORY_FILE_PATH = memory_file_path
        self._number_files = 0
        
        self._MAX_TIME = MAX_TIME
        self._realized_rewards = 0.
        self._implied_rewards = 0.
        self._implied_reward_bias = 0.
        self._REALIZED_REWARDS_HORIZON = 20
        self._implied_realized_reward_ratio = []

        # Create networks for image compression and pre-processing
        self._image_in, self._image_out = self._reduce_image_graph()
        self._input_layer = tf.placeholder("float", 
                [None, RESIZED_SCREEN, RESIZED_SCREEN, STATE_FRAMES])

        # Create the graph architecture
        # In order to preserve flexibility, all layers are appended to lists

        # Convolutional layers
        self._convolutional_weights = []
        self._convolutional_bias = []
        self._hidden_convolutional_layer = []
        self._hidden_max_pooling_layer = []
        self._dropout_prob_keep = tf.placeholder('float')
        self._dropout_layer = [self._input_layer]

        for i in range(len(CONVOLUTION_FILTER_VECTOR)):
            self._convolutional_weights.append(tf.Variable(tf.truncated_normal(
                [CONVOLUTION_FILTER_VECTOR[i], CONVOLUTION_FILTER_VECTOR[i], 
                 CONVOLUTION_INPUT_VECTOR[i], CONVOLUTION_KERNEL_VECTOR[i]], 
                mean=INITIALIZATION_MEAN, stddev=INITIALIZATION_STDDEV)))
            self._convolutional_bias.append(tf.Variable(tf.constant(
                INITIALIZATION_BIAS, shape=[CONVOLUTION_KERNEL_VECTOR[i]])))
            self._hidden_convolutional_layer.append(
                self._activation(tf.nn.conv2d(
                    self._dropout_layer[i], 
                    self._convolutional_weights[i], 
                    strides=[1, CONVOLUTION_STRIDE_VECTOR[i],
                        CONVOLUTION_STRIDE_VECTOR[i], 1], 
                    padding="VALID") + self._convolutional_bias[i])
                    )

            if MAXPOOL_VECTOR[i] > 1:
                self._hidden_max_pooling_layer.append(tf.nn.max_pool(
                    self._hidden_convolutional_layer[i],
                    ksize=[1, MAXPOOL_VECTOR[i], MAXPOOL_VECTOR[i], 1],
                    strides=[1, MAXPOOL_VECTOR[i], MAXPOOL_VECTOR[i], 1], 
                    padding="SAME"))
            else:
                self._hidden_max_pooling_layer.append(
                    self._hidden_convolutional_layer[i])
            
            if self._dropout_prob_keep < 1.:                
                self._dropout_layer.append(tf.nn.dropout(
                    self._hidden_max_pooling_layer[i], 
                    self._dropout_prob_keep)
                    )
            else:
                self._dropout_layer.append(self._hidden_max_pooling_layer[i])

        # Feed forward layers
        self._hidden_activation_layer = [tf.reshape(
            self._dropout_layer[-1], [-1, FEED_FWD_VECTOR[0]])]
        self._feed_forward_weights = []
        self._feed_forward_bias = []

        for i in range(len(FEED_FWD_VECTOR) - 2):
            self._feed_forward_weights.append(tf.Variable(tf.truncated_normal(
                [FEED_FWD_VECTOR[i], FEED_FWD_VECTOR[i+1]], 
                mean=INITIALIZATION_MEAN, stddev=INITIALIZATION_STDDEV)))
            self._feed_forward_bias.append(tf.Variable(tf.constant(
                INITIALIZATION_BIAS, shape=[FEED_FWD_VECTOR[i+1]])))
            self._hidden_activation_layer.append(
                self._activation(
                    tf.matmul(self._hidden_activation_layer[i], 
                              self._feed_forward_weights[i]) 
                    + self._feed_forward_bias[i])
                    )
                    
        # The calculation of the state-action value function does not 
        # require the neurons' activation function
        self._feed_forward_weights.append(tf.Variable(tf.truncated_normal(
            [FEED_FWD_VECTOR[-2], FEED_FWD_VECTOR[-1]], 
            mean=INITIALIZATION_MEAN, stddev=INITIALIZATION_STDDEV)))
        self._feed_forward_bias.append(tf.Variable(tf.constant(
            INITIALIZATION_BIAS, shape=[FEED_FWD_VECTOR[-1]])))
        self._state_value_layer = (tf.matmul(
            self._hidden_activation_layer[-1], self._feed_forward_weights[-1]) 
            + self._feed_forward_bias[-1])

        # Define the logic of the optimization
        self._action = tf.placeholder("float", [None, NUMBER_ALLOWED_ACTIONS])
        self._target = tf.placeholder("float", [None])
        self._action_value_vector = tf.reduce_sum(tf.mul(
            self._state_value_layer, self._action), reduction_indices=1)
        self._cost = tf.reduce_sum(tf.square(
            self._target - self._action_value_vector))
        self._alpha = tf.placeholder('float')
        self._train_operation = tf.train.AdamOptimizer(
            self._alpha).minimize(self._cost)
        self._session = tf.Session()

        operation_intizializer = tf.initialize_all_variables()
        self._saver = tf.train.Saver()

        try:
            self._saver.restore(self._session, self._PARAMETERS_FILE_PATH)
            print ('Calibrated parameters SUCCESSFULLY LOADED.',
                   'Agent is ready to play.', flush=True)
            self._is_training = False
        except:
            self._session.run(operation_intizializer)
            print ('It was not possible to load calibrated parameters.',
                   'Agent will undergo training.', flush=True)
            self._is_training = True

        if FORCED_TRAINING:
            self._is_training = FORCED_TRAINING
            
        if self._is_training:
            try:
                for index, file in enumerate(self._MEMORY_FILE_PATH):
                    memory_in_file = np.load(file)
                    print('File', file, 'loaded', flush=True)
                    for elem in memory_in_file:
                        self._previous_observations.append(elem)
                del memory_in_file
                print('Previous observations loaded.',
                      'Training can resume at past pace.', flush=True)
                print ('Length of training data loaded from file:', 
                       len(self._previous_observations), flush=True)
            except:
                print('Previous observations could not be loaded.',
                      'This will slow down further training.',
                      flush=True)
        self._time = 0


    def _activation(self, z, a=0.01):
        """Note that I used the Leaky ReLU function for neural activation. 
        The simple ReLU function led to many neurons dying during training, 
        and the other options slowed training too much. 
        """
        return (tf.maximum(a*z, z)) 
        # return (tf.nn.relu(z))
        # return (tf.sigmoid(z))
        # return (tf.tanh(z))
                    
    def _reduce_image_graph(self):
        # Graph for compressing the input image into an image of 
        # shape=[RESIZED_SCREEN,RESIZED_SCREEN]
        image_input_layer = tf.placeholder("float", 
            [None, self._FIELD_OF_VIEW, self._FIELD_OF_VIEW, 1])
        image_step_size = int((self._FIELD_OF_VIEW / RESIZED_SCREEN))
        image_output_layer = tf.nn.max_pool(
            image_input_layer, 
            ksize=[1, image_step_size, image_step_size, 1],
            strides=[1, image_step_size, image_step_size, 1], 
            padding="SAME")                                         
        return (image_input_layer, image_output_layer)         
        
    def _compress_image(self, image):
        # Compress the input image into a pre-set format
        compressed_image = self._session.run(
            self._image_out, 
            feed_dict={self._image_in: np.array([image[:,:,COLOUR:COLOUR+1]])}
            )
        compressed_image = (
            compressed_image[:,:RESIZED_SCREEN,:RESIZED_SCREEN,:]/255.)     
        return (compressed_image.T)
    
    
    def choose_action(self, image, reward, is_terminal):
        '''This method is the interface between the Agent and the game
            It processes the raw pixels, the reward received at that state,
            and whether it is a terminal state or not, and returns a decision
            '''
            
        # Update the agent's clock
        self._time += 1
        
        # Update self._realized_reward. This is not used for training, 
        # but to verify (after training) how accurate the estimate of the
        # action-value function is vs the actually realized discounted rewards.
        # Note that the parameters of the action-value function were calibrated 
        # for the below game settings, and should you change the settings the 
        # fair value of the function should change.
        # PROB_OBJECT_SPAWNED = 0.12   
        # PROB_GOLD_SPAWNED = 0.8   
        self._realized_rewards /= DISCOUNTING_RATE
        self._realized_rewards += reward

        # Compress the input image into a pre-set format
        compressed_image = self._compress_image(image)

        # If the CNN has no last state, fill it by using the current state,
        # choose a random action, and return the action to the game
        if len(self._last_state) == 0:
            self._last_state = np.stack((
                compressed_image for _ in range(STATE_FRAMES)),
                axis=3)[:, :, :, :, 0]
            self._last_action[np.random.randint(0, NUMBER_ALLOWED_ACTIONS)] = 1            
            return self._decision_from_action_vector(self._last_action)

        if self._time == STATE_FRAMES:
            # Measure the value of the starting state
            # Note this is dependent on the calibration and the game settings
            # during training
            self._implied_reward_bias = np.max(
                self._session.run(
                       self._state_value_layer, 
                       feed_dict={self._input_layer: self._last_state, 
                                  self._dropout_prob_keep: 1.}
                        )
                )
                
        # Update the current state 
        # current_state is made by (STATE_FRAMES) reduced images
        current_state = np.append(compressed_image, 
            self._last_state[:, :, :, :-1], axis=3)

        # Append the current observation (which includes previous state,
        # action,  reward, and new state) to the memory only for some of the 
        # observed instances
        if reward or is_terminal or PROB_KEEP_IN_MEMORY_IF_UNEVENTFUL:
            new_observation = [0 for _ in range(5)]
            new_observation[LAST_STATE_IDX] = self._last_state.copy()
            new_observation[ACTION_IDX] = self._last_action.copy()
            new_observation[REWARD_IDX] = reward
            new_observation[CURR_STATE_IDX] = current_state.copy()
            new_observation[TERMINAL_IDX] = is_terminal
            self._previous_observations.append(new_observation)
            
        # If the memory is full, save part of it into file  
        if len(self._previous_observations) >= REPLAY_MEMORY:
            temp_copy = []
            while self._previous_observations:
                temp_copy.append(self._previous_observations.pop())

            file_copy = []
            for _ in range(0, MAX_OBSERVATIONS_IN_FILE):
                file_copy.append(temp_copy.pop())
            while temp_copy:
                self._previous_observations.append(temp_copy.pop())
                
            np.save('obs_' + str(self._number_files) + '.npy', file_copy)
            del temp_copy, file_copy
            self._number_files += 1

        # Run the training at fixed intervals
        if (self._is_training and 
            len(self._previous_observations) > MIN_MEMORY_FOR_TRAINING and 
            self._time % TRAINING_STEPS == 0):
                verbose = not (self._time % (500 * TRAINING_STEPS))
                self._train(verbose=verbose)       

        # Update variables before the end of function
        self._last_state = current_state.copy()

        # Choose next action
        self._last_action = self._make_decision_on_next_action()

        return self._decision_from_action_vector(self._last_action)


    def _train(self, alpha=ALPHA, verbose=False):
        # Method that runs the Agent's training 
        
        # Sample a mini_batch to train on
        permutations = np.random.permutation(
            len(self._previous_observations))[:MINI_BATCH_SIZE] 
        previous_states = np.concatenate(
            [self._previous_observations[i][LAST_STATE_IDX][:,:,:,:STATE_FRAMES] 
            for i in permutations], 
            axis=0)
        actions = np.concatenate(
            [[self._previous_observations[i][ACTION_IDX]] 
            for i in permutations], 
            axis=0)
        rewards = np.array(
            [self._previous_observations[i][REWARD_IDX] 
            for i in permutations]).astype('float')
        current_states = np.concatenate(
            [self._previous_observations[i][CURR_STATE_IDX][:,:,:,:STATE_FRAMES]  
            for i in permutations], 
            axis=0)  
        is_terminal = np.array(
            [self._previous_observations[i][TERMINAL_IDX] 
            for i in permutations]).astype('bool')

        # CNN calculates the value of the current states-action pairs
        value_of_current_states_vs_actions = self._session.run(
            self._state_value_layer, 
            feed_dict={self._input_layer: current_states, 
                       self._dropout_prob_keep: 1.})

        # The value of the best state-action pair is assigned to the value of 
        # the current state, hence the estimated value of the current state is 
        # not affected by (sub-optimal) choices taken during exploration
        estimated_value_of_states = rewards.copy()
        estimated_value_of_states += ( 
            (1. - is_terminal) * 
            DISCOUNTING_RATE * 
            np.max(value_of_current_states_vs_actions, axis=1)
            )

        # Train the CNN: link the actions taken in the previous states to 
        # the rewards observed in the new state + 
        # the (estimated) value of the new state
        self._session.run(
            self._train_operation, 
            feed_dict={self._input_layer: previous_states,
                       self._action: actions,
                       self._target: estimated_value_of_states,
                       self._dropout_prob_keep: DROPOUT_PROBABILITY,
                       self._alpha : alpha})        

        # Tools for monitoring the learning process
        if verbose: 
            print ('Epoch:', int(self._time/TRAINING_STEPS), flush=True)        
            print ('Values of actions (pre) ', 
                   value_of_current_states_vs_actions[-1], 
                    flush=True)
            print ('Values of actions (post)', 
                   self._session.run(
                       self._state_value_layer, 
                       feed_dict={self._input_layer: current_states, 
                                  self._dropout_prob_keep: 1.}
                        )[-1], 
                    flush=True)


    def _make_decision_on_next_action(self):        
        '''Method returns next action, which is chosen either greedly (ie 
            relying on the CNN estimate) or randomly (ie as part of exporation)
            '''
        epsilon = np.max([EPSILON_EXPLORATION_AT_REGIME,
                1. - 
                np.max([0., self._time - MIN_MEMORY_FOR_TRAINING])/
                (TIME_EXPLORATION_AT_REGIME - MIN_MEMORY_FOR_TRAINING)*
                (1. - EPSILON_EXPLORATION_AT_REGIME)
                 ])            
         
        # Calculation of the value of different actions
        cnn_value_per_action = self._session.run(
            self._state_value_layer,
            feed_dict={self._input_layer: self._last_state, 
                       self._dropout_prob_keep: 1.})
                       
        # CNN choses the next action with an epsilon-greedy approach
        if self._is_training and np.random.random() < epsilon:
            chosen_action_index = np.random.randint(0, NUMBER_ALLOWED_ACTIONS)
        else:
            chosen_action_index = np.argmax(cnn_value_per_action[0])

        # Store the ratio between the expected rewards implied by the CNN and
        # the realized (discounted) rewards received while implementing the
        # strategy
        if (not self._is_training and 
            self._time % self._REALIZED_REWARDS_HORIZON == 0):
            self._realized_rewards *= (DISCOUNTING_RATE ** 
                                        self._REALIZED_REWARDS_HORIZON)
            ratio = (self._implied_rewards / 
                        (self._realized_rewards + 
                        self._implied_reward_bias))
            MA = 50
            self._implied_realized_reward_ratio.append(ratio)
            print('K-step implied/realized reward ratio:', 
                  round(ratio,2),
                  '- Average of last 50 measurements:', 
                  round(np.mean(self._implied_realized_reward_ratio[-MA:]),2),
                  flush=True)
            self._implied_rewards = np.max(cnn_value_per_action[0])
            self._realized_rewards = 0.    

        next_action_vector = np.zeros([NUMBER_ALLOWED_ACTIONS])
        next_action_vector[chosen_action_index] = 1
        return next_action_vector


    def _decision_from_action_vector(self, action):
        # Convert the CNN chosen action into a format accepted by the game
        action_legend = {0: -1,
                         1: 0,
                         2: +1}               
        return action_legend[np.argmax(action)]


    def close(self):

        # If training, save the RAM memory to file
        if self._is_training:
            self._saver.save(self._session, self._PARAMETERS_FILE_PATH)
            
            temp_copy = []
            while self._previous_observations:
                temp_copy.append(self._previous_observations.pop())

            while len(temp_copy) > MAX_OBSERVATIONS_IN_FILE:   
                file_copy = []
                for _ in range(0, MAX_OBSERVATIONS_IN_FILE):
                    file_copy.append(temp_copy.pop())
                np.save('obs_' + str(self._number_files) + '.npy', file_copy)
                self._number_files += 1   
                del file_copy                 
            
            np.save('obs_' + str(self._number_files) + '.npy', temp_copy)
            print('\nTotal number of saved file containing transitions:',
                  self._number_files + 1)
            del temp_copy
   
        # Close the session and clear TensorFlow's graphs             
        ops.reset_default_graph() 
        self._session.close()
        
        # Plot the graph of the average Implied/Realized reward ratio
        MA = 50       # moving average parameter
        if len(self._implied_realized_reward_ratio) > MA:
            plt.figure()
            plt.subplot(111)
            title = plt.title(('History of implied/realized reward ratio'), 
                                fontsize="x-large")
            line = self._implied_realized_reward_ratio.copy()
            average = [np.mean(line[i:i+MA]) for i in range(len(line)-MA)]
            dt = [i for i in range(len(line))]
            plt.plot(dt, line, 'r', 
                     label='Individual observations')
            plt.plot(dt[MA:], average, 'b', 
                     label='50-observation moving average')
            plt.axis([0, len(line), np.min(average)*0.5, np.max(average)*2.])
            title.set_y(1.0)
            plt.legend()
            plt.show()

    def _visualize_layer(self, input_image, path='saved_image_for_testing'):
        # Helps visualizing what different layers see and what the filters are
    
        image = input_image.copy()        
        if len(image.shape) < 3: 
            print('There is an issue with the shape of the input image')
        if len(image.shape) == 4 and image.shape[0] == 1: 
            image = image[0,:,:,:]
        elif len(image.shape) == 4 and image.shape[2] == STATE_FRAMES:
            image = image[:,:,0,:]
        x1, x2, x3 = image.shape
        
        gap = 1
        grey_scale_image = 255.* np.ones([x1+2*gap, (x2+gap)*x3+gap, 1])
        for i in range(0, x3):
            if image[:, :, i:i+1].max() > 0:
                max_coeff = 255./(image[:, :, i:i+1].max()) 
            else:
                max_coeff = 0
            grey_scale_image[gap:-gap, (i+1)*gap+x2*i:(i+1)*gap+x2*(i+1), 
                :1] = image[:, :, i:i+1]*max_coeff

        for i in range(x1):
            for j in range(x3*(x2+1)):
                grey_scale_image[i, j, 0] = int(grey_scale_image[i, j, 0])
                
        grey_scale_image = np.append(grey_scale_image[:,:,:], 
                                     np.append(grey_scale_image[:,:,:], 
                                               grey_scale_image[:,:,:], 
                                               axis=2), 
                                    axis=2)
                                    
        # Save the visualized layer to file 
        plt2.image.imsave(path + '_.jpg', grey_scale_image)   


