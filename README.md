# Overview

This is a deep learning environment for training bots on the game wordle. Why have fun when you can let your computer have fun for you!
## Wordle

https://www.nytimes.com/games/wordle/index.html

Wordle is a word game where you have 6 tries to guess a hidden 5 letter word. Each guess you make gives you information about what the hidden word is. There are 3 results possible for each letter: 
1. the letter is in the word AND in the right spot. 
2. the letter is in the word but not in the right spot.
3. the letter is not in the word

Based on this information, if you choose your words wisely, you can deduce the hidden word.

## How to use this repo

**Warning**
Ray can be super fidly with how you want the workers spread across self-play, training etc.
Tweak the following parameters in config to get the desired result.
- self_play_delay
- training_delay
- num_workers

Repo setup

1. run ```python setup.py```
2. install requirements.txt

#### Tests

run tests `python -m pytest tests`

#### Dynamics Function Games Generation

generate games for dynamics function pre-training

#### Dynamics Function Pretraining

There are two flavors for pretraining the dynamics function
I've found that ray doesn't always ultilize my gpus like i would want, so i implemented dynamics training with PyTorch DDP as well as with Ray.
- train_dynamics_ddp.py - This is parallelized with pytorch DDP
- train_dynamics_ray.py - This is parallelized with ray

- test_dynamics.py - 
- test_policy.py - 


- main.py contains the following abilities
    - train the actor, critic and dynamics function via self play
    - load a model
    - validate the model manually (can check model outputs via command line)
    - validate MCTS. validate the MCTS tree. Outputs a pdf
    - validate the replay buffer. Check your model inputs to make sure the representations are correct.
    - validate model updates. Step through the model learning process.

#### Frontend

There is a small frontend for this project located [here](https://github.com/Morgan-Griffiths/wordle_frontend)

To show the bot in action:
1. Train a version with main.py
2. Store your desired weights/date-time/model.checkpoint as weights/production.checkpoint
3. Run `python server.py`
4. Switch to wordle_frontend and run `python app.py`
5. Navigate to localhost:5000
6. Watch your bot take its first steps!

## Strategy

Fundamentally this is a game about information. There is a hidden state - which is the actual word. We know it is a 5 letter work (it is actually 1 of ~2400 possible 5 letter words).
Initially our prior probability of every word being the answer is equal. After we make a guess, ideally we have eleminated many of the available words. You can think of this as collapsing the distribution. Once we have collapsed the distribution to 1 word remaining we know exactly what it is.

Additionally everytime you play it, the hidden word will be different. This makes the game non-stationary. This means that the word we discovered the previous game has no baring on subsequent games. So each game is starting afresh. 

In Reenforcement Learning (RL), this is known as an imperfect information non-stationary game.

A nice heuristic for playing Wordle is to maximize the amount of information gained per guess. This is not necessarily optimal because maximizing information on a given round, may not maximize information gained over 2 or more rounds. 

For this agent however, we will simply be optimizing for the end result. Which will also solve the game in optimum average speed.

## The RL Agent

My approach here is modeled off of DeepMind's MuZero. It builds off previous work from DeepMind with AlphaGo and AlphaZero. A quick summary of MuZero is as follows:

1. It builds an internal representation of the environment. This is called the 'Dynamics Function'. This function predicts the next state S', from the current state S, given action A. It also learns the reward function to predict the reward R.
2. It has an encoder which takes the state S -> turns it into a hidden state H. Which is the input to the dynamics function.
3. It has an Actor or policy network, which produces probabilities over actions given the current state S
4. It has a Critic or value network, which produces the value of the current state V.
5. As MuZero searches the tree, at each state the policy is queried and the probabilities are used to help select the next action. In addition there is the UCB (Upper confidance bound) algorithm which balances between exploration and exploitation. 

In essence there are 4 things at play.
1. State encoder
2. Dynamics function
3. Policy
4. Critic

You can read the paper [here](https://arxiv.org/pdf/1911.08265.pdf).

### How my implementation differs

A key aspect of MuZero is its use in perfect information games. In such games, the dynamics function is deterministic. For example if i show you a chess board and tell you what my next move is. You know exactly what the next board position will be. This is not the case in Wordle.

Because of this, instead of outputting S', i have the dynamics function output a probability distribution over next states. And then during tree search we sample from the distribution to get S'. This complicates the MCTS function a bit because now we have two types of nodes - state nodes which store action probabilities, and action nodes which store state probabilities.

**Encoder**
I dispense with the encoder because the wordle transition function is quite simple. And have the network iterate over actual game states. Also in a Typical MuZero algorithm it can search beyond the end of the game. Because Wordle is at MOST 6 rounds, that is unnecessary here.

**The dynamics function** 
Outputs a probability distribution over results. There are 243 (5 squares, 3 possible outcomes for each square, 5^3) possible results. I save 0 for padding, so the results range from 1-244. winning game state is 244.

**Reward function**
- win           +1 
- no result      0  
- loss          -1 
The reward function: 
- 0 can happen on the first 5 turns
- 1 can happen on any turn
- -1 can only happen on the last turn

Because there is only one vector [3,3,3,3,3] that corresponds to +1 ALWAYS, everything else is either 0 or -1 depending on whether it is the last turn or not. This means we don't need to use a NN to predict the reward, we can hardcode the reward and construct the reward distribution based on a boolean output that predicts whether its the last turn or not. 
Then when we sample S' we will index the corresponding reward.

## Training

Learning the state transition function does not rely on the actor. This is because the goal of the dynamics function is the learn the transition function for any action in any state. Because of this, we can train the dynamics function first on randomly sampled games. This is a huge time saver, because the policy is only as good as the dynamics function. If the dynamics function erroneously reports that we have lost the game, even though we choose the right action, the policy will be updated improperly.

Therefore, it is recommended to throughly train the dynamics function first. And then train the actor and critic.
## Encodings

To make wordle machine digestible we have to convert the game state into numbers. 

Each letter is encoded via the following format. With 0 reserved for padding

| Letter      | Number |
| ----------- | ----------- |
| _ | 0|
| a | 1|
| b | 2|
| c | 3|
| d | 4|
| e | 5|
| f | 6|
| g | 7|
| h | 8|
| i | 9|
| j | 10|
| k | 11|
| l | 12|
| m | 13|
| n | 14|
| o | 15|
| p | 16|
| q | 17|
| r | 18|
| s | 19|
| t | 20|
| u | 21|
| v | 22|
| w | 23|
| x | 24|
| y | 25|
| z | 26|

Each letter in the word has an associated result.

| Result      | Meaning |
| ----------- | ----------- |
| 0 | padding|
| 1 | Missing (letter not in word)|
| 2 | Contained (letter present in word, but in a different position)|
| 3 | Exact (letter present and in that exact location)|

If we take the following image

![img](./images/wordle.png "Wordle")

letter encoding will be
[5,14,1,3,20]
result encoding will be
[2,1,3,3,2]

# Credits

Many thanks to 3b1b and werner-duvaud for some code and inspiration.

- https://github.com/3b1b/videos/tree/master/_2022/wordle
- https://github.com/werner-duvaud/muzero-general