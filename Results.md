# Outline

Develop and train MuZero on the Wordle environment.

# Overview of wordle

# MuZero

**Perfect Information**
Muzero consists of 3 parts:
- Policy - given S outputs A
- Encoder - given S outputs h the encoded state
- Dynamics function - given (S,A) output S' and R. 

In order to adapt Muzero to an imperfect information space, we need to change the dynamics function to a probability distribution over S'. 
**Imperfect Information**
- Dynamics function - given (S,A) output P(S') and P(R). 


In doing MCTS search, we will sample the state probability distribution to get a guess at what the next state will be. By sampling this many times, we will get a good idea of next states, relative to the accuracy of the probability distribution.

## Tests

We must assert the following:
- that we can learn the probability distributions of P(S') and P(R).
- The Policy can output the correct choice given (S,A). I.E. The network can process the representation and has sufficient representational capacity to choose A

## Architecture

### Dynamics function

**State Transition function**
there are 243 (5 squares, 3 possible outcomes for each square, 5^3) possible results. 
- 0 padding
- 1 missing
- 2 contained
- 3 exact
the vector [3,3,3,3,3] corresponds to getting every letter in its correct position.
Because the actual probability function will be quite difficult to make, i expect pretraining the state transition function will make everything much easier. Then we can check the validity of the trained model. If everything is good, training the Policy will be the next step.


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

State transition function can be trained exhaustively on random samples of the game.

Sample game outcomes:
- update P(S')

after the end of game, propogate the reward back in time, via the discount factor \lambda.

## Issues

Wordle is an imperfect information game. The states probabilistically map to each other. Traditionally MuZero is only applicable to perfect information games, because it predicts the mapping of (S,A) -> S'. In the current case, the mapping is not to one state but to a distribution over states. 