# player-ConvNN
**Implementation of a Convolutional Neural Network that applies the deep Q-learning algorithm to play PyGame video games.**  
Specifically, I intended to replicate DeepMind's [paper][3] for a simple arcade game from my GitHub page, [Asteroids][1]. I ended up making slight variations to DeepMind's implementation (see more below).  
The interesting bit of the code is that the game's state is expressed in raw pixel data, whose value estimation requires a CNN. Check out the [Youtube video][8] that shows the CNN's behaviour under different degrees of difficulty.  

As inputs, it accepts:
- `image`: array, image made of raw pixels captured from the game screen;  
- `reward`: float, reward received at that particular state;  
- `is_terminal`: bool, indication of whether the player is at a terminal state;  

and returns:
- an `integer`, as the chosen action.

### Quick start
Download the files `CNN_Agent.py` and `Parameters_CNN.ckpt` to the same folder as the game `asteroids v1.0.py`, which can be found [here][2].  
Then, launch the game `asteroids v1.0.py` and enjoy watching the AI playing it.  

The below is a brief instance of the game. Notice that to get to this level the CNN only requires 150,000 frame observations, i.e. approx 5 hours of playing.  
<img src="https://github.com/FlankMe/player-ConvNN/blob/master/images/instance.gif" width="50%" />

In here, I modified the settings of the videogame to make it very difficult, even for a human. The agent appears to be doing very well anyway (certainly better than I would do!).  
<img src="https://github.com/FlankMe/player-ConvNN/blob/master/images/instance_hard_settings.gif" width="50%" />
 
### Optimal strategy vs State-Action value function
One of the challenges of the project was to measure the algorithm's performance.  
I couldn't come up with any way to compare the agent's performance to the optimal strategy (assuming it exists). I could only observe its performance vs humans.   

However, the bit that never stops impressing me is that Q-learning (among other RL methods) doesn't merely approximate the optimal strategy, but attempts to estimate the **discounted value of future rewards for each decision**.  
The below graph shows the 50-step moving average of the ratio between estimated future rewards and realized discounted rewards obtained pursuing the strategy. The fact that the average ratio gravitates around `1.0`, gives me confidence that the agent is accurately estimating the value of its strategy.

<img src="https://github.com/FlankMe/player-ConvNN/blob/master/images/performance.jpeg" width="60%" />

### Pseudocode
The agent adopts the deep-Q learning algorithm: the state-action-value function is calibrated by minimizing (for small minibatches of previous observations) the below cost function 
```sh
( [r_t+1 + max_a {Q(s_t+1,a)}]  -  Q(s_t,a_t) )^2  
```
where `Q(s_t, a_t)` is the value associated with taking action `a_t` on state `s_t`, and `r_t+1` is the reward observed at time `t+1`.

More specifically, the algorithm involves:  
*Initialize the agent's action-value* Q *function with random weights;*  
**_While_** *running:*  
*-- Take an epsilon-greedy decision on what to do at the current state;*  
*-- Observe the following reward and screen's image;*  
*-- Convert the observed image to grey scale and compress it to a pre-determined smaller format* [I chose 84x84]*;*  
*-- Store the transition from the current state to the (now observed) following state and following reward;*  
*-- Run a step of the learning algorithm, ie select a minibatch of transitions from previous observations* [I chose size 32] *and take one optimization step by minimizing the above cost function* [I chose AdamOptimizer]*;*  

### Key differences with DeepMind's architecture
I chose to use:  
- a different **activation function**: I used the leaky ReLU function, defined as `max(a*z, z)` with `a << 1`, as opposed to the simple ReLU function defined as `max(0, z)`. The reason for this was that the ReLU function led to many neurons "dying" during training as they got stuck in permanently negative territory and became impossible to train further. It took me long time to identify this issue;  
- a different **network**: this CNN is deeper (3 convolutional layers), but uses less parameters (smaller filters and less kernels) as well as a smaller feed forward hidden layer;  
- a more **flexible** graph that allows to easily add *maxpooling layers* and an implementation of the *dropout technique*. In fairness, I ended up using none of these extra features as they didn't seem to any value.  

### Visual processing
Before feeding the screen's image to the CNN, the input is pre-processed first: the image is converted to grey scale, compressed to a smaller resolution, and stacked to previous frames [*I chose to feed the CNN with the 3 most recent frames stacked together*].  

Here I show an example of an input image and its resulting (compressed) stacked images fed to the CNN:  

<img src="https://github.com/FlankMe/player-ConvNN/blob/master/images/screen_s_snapshot.jpg" width="60%" />  

<img src="https://github.com/FlankMe/player-ConvNN/blob/master/images/pre-processed_state.jpg" width="60%" />  

Out of interest, here is the visualization of the first-layer filters of the CNN.  
Contrary to other applications, understanding what the first-layer filters do is not intuitive at all.  
<img src="https://github.com/FlankMe/player-ConvNN/blob/master/images/filters.jpg" width="160%" />

### Resources & Acknowledgements
* [Playing with Atari with Deep Reinforcement Learning][3], by DeepMind Technologies;  
* Daniel Slater's [blog][4], and in particular his PyGamePlayer [code][5] that I used as starting point for mine.

### Requirements
* **Python 3**. I recommend this version as it's the only one I found compatible with the below libraries;
* **PyGame**, I used version 1.9.2a0. Download it from [here][6];
* **TensorFlow**, I only managed to install it on my Mac. Download it from [here][7];
* **Asteroids**, arcade game from my GitHub page, whose code can be found [here][2].

[1]: https://github.com/FlankMe/Asteroids
[2]: https://github.com/FlankMe/Asteroids/blob/master/asteroids%20v1.0.py
[3]: https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf   
[4]: http://www.danielslater.net/2016/03/deep-q-learning-pong-with-tensorflow.html
[5]: https://gist.github.com/DanielSlater/2b9afcc9dfa6eda0f39d#file-create-network
[6]: http://www.pygame.org/download.shtml
[7]: https://www.tensorflow.org/versions/r0.7/get_started/os_setup.html
[8]: https://youtu.be/cPm6IbHtDZs
