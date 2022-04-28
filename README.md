# Wordle

deep learning environment for training bots on the game wordle.

# test

```pytest```

# train

```python main.py``` 

on menu prompt enter 0

# run experiments

```python run_experiments.py```

# weights

Saved network weights

# results

Tensorboard stats

# plots

are in assets

# MuZero in wordle

Wordle is a non stationary, imperfect information game. Traditionally muzero maps (s,a) -> s' in a deterministic fashion. However, in non-stationary, imperfection informations games, we must map (s,a) -> distribution over s'.
In wordle, given a state s, and action (5 letter word) a, there is a 243 length distribution (3^5) over a (5,3) result vector. 5 letters [0,27] and 3 states [0,4]. With a padding dimension at 0. Then sample the 243 length vector and combine with s to make s'. 

In the same way an algorithm can be made, to maximize the collapse of the probability distribution function. given a state s and available words w, pick the word w that collapses the distribution function the most. At the limit, we will collapse the distribution function down to 1 possibility, when we have all the necessary information to pick the word.