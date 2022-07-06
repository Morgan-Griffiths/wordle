from globals import dictionary_index_to_word,Embeddings,alphabet_dict
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

index_batch = [[162, 4], [521, 1], [366, 1], [659, 4]]

state_batch = torch.tensor([[[[12,  1,  5],
          [15,  2,  5],
          [21,  1,  5],
          [18,  2,  5],
          [26,  1,  5]],

         [[12,  1,  5],
          [15,  2,  5],
          [21,  1,  5],
          [18,  2,  5],
          [26,  1,  5]],

         [[ 5,  1,  7],
          [13,  1,  7],
          [ 5,  1,  7],
          [20,  2,  7],
          [18,  2,  7]],

         [[ 1,  1,  0],
          [ 1,  1,  0],
          [ 8,  1,  0],
          [ 5,  1,  0],
          [ 4,  1,  0]],

         [[ 0,  0,  0],
          [ 0,  0,  0],
          [ 0,  0,  0],
          [ 0,  0,  0],
          [ 0,  0,  0]],

         [[ 0,  0,  0],
          [ 0,  0,  0],
          [ 0,  0,  0],
          [ 0,  0,  0],
          [ 0,  0,  0]]],


        [[[12,  1,  5],
          [15,  3,  5],
          [21,  1,  5],
          [18,  1,  5],
          [26,  1,  5]],

         [[ 0,  0,  0],
          [ 0,  0,  0],
          [ 0,  0,  0],
          [ 0,  0,  0],
          [ 0,  0,  0]],

         [[ 0,  0,  0],
          [ 0,  0,  0],
          [ 0,  0,  0],
          [ 0,  0,  0],
          [ 0,  0,  0]],

         [[ 0,  0,  0],
          [ 0,  0,  0],
          [ 0,  0,  0],
          [ 0,  0,  0],
          [ 0,  0,  0]],

         [[ 0,  0,  0],
          [ 0,  0,  0],
          [ 0,  0,  0],
          [ 0,  0,  0],
          [ 0,  0,  0]],

         [[ 0,  0,  0],
          [ 0,  0,  0],
          [ 0,  0,  0],
          [ 0,  0,  0],
          [ 0,  0,  0]]],


        [[[12,  2,  5],
          [15,  1,  5],
          [21,  1,  5],
          [18,  1,  5],
          [26,  3,  5]],

         [[ 0,  0,  0],
          [ 0,  0,  0],
          [ 0,  0,  0],
          [ 0,  0,  0],
          [ 0,  0,  0]],

         [[ 0,  0,  0],
          [ 0,  0,  0],
          [ 0,  0,  0],
          [ 0,  0,  0],
          [ 0,  0,  0]],

         [[ 0,  0,  0],
          [ 0,  0,  0],
          [ 0,  0,  0],
          [ 0,  0,  0],
          [ 0,  0,  0]],

         [[ 0,  0,  0],
          [ 0,  0,  0],
          [ 0,  0,  0],
          [ 0,  0,  0],
          [ 0,  0,  0]],

         [[ 0,  0,  0],
          [ 0,  0,  0],
          [ 0,  0,  0],
          [ 0,  0,  0],
          [ 0,  0,  0]]],


        [[[20,  1,  9],
          [18,  1,  9],
          [15,  2,  9],
          [20,  1,  9],
          [19,  1,  9]],

         [[12,  1,  5],
          [15,  3,  5],
          [21,  1,  5],
          [18,  1,  5],
          [26,  1,  5]],

         [[12,  1,  5],
          [15,  3,  5],
          [21,  1,  5],
          [18,  1,  5],
          [26,  1,  5]],

         [[ 8,  1,  4],
          [15,  3,  4],
          [22,  1,  4],
          [ 5,  1,  4],
          [ 1,  3,  4]],

         [[ 0,  0,  0],
          [ 0,  0,  0],
          [ 0,  0,  0],
          [ 0,  0,  0],
          [ 0,  0,  0]],

         [[ 0,  0,  0],
          [ 0,  0,  0],
          [ 0,  0,  0],
          [ 0,  0,  0],
          [ 0,  0,  0]]]])

action_batch = torch.tensor([2, 5, 1, 0])

value_batch = torch.tensor([[-0.7225],
        [-0.4437],
        [-0.4437],
        [-0.7225]])

reward_batch = torch.tensor([[0.],
        [0.],
        [0.],
        [0.]])

policy_batch = torch.tensor([[0.0000, 0.0000, 1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000],
        [0.4000, 0.0000, 0.0000, 0.0000, 0.0000, 0.4000, 0.0000, 0.0000, 0.0000,
         0.2000],
        [0.0000, 0.6000, 0.0000, 0.0000, 0.2000, 0.0000, 0.0000, 0.0000, 0.2000,
         0.0000],
        [0.2000, 0.0000, 0.0000, 0.0000, 0.6000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.2000]])

weight_batch = [0.93172336,0.95254445,0.91430515,1.]

result_batch = torch.tensor([ 0, 54,  1, 81])

gradient_scale_batch = [[2], [5], [5], [2]]



# create index to int letters
B = state_batch.shape[0]
action_letters = torch.tensor(np.stack([np.array([alphabet_dict[letter] for letter in dictionary_index_to_word[a.item()]]) for a in action_batch])) # (B,5)
state_letters = state_batch[:,:,:,Embeddings.LETTER] # (B,6,5)
state_results = state_batch[:,:,:,Embeddings.RESULT] # (B,6,5)

print('action_letters',action_letters,action_letters.shape)
print('state_letters',state_letters,state_letters.shape)
print('state_results',state_results.shape)
print('previous_actions',previous_actions.shape)
print('state_letters[:,:,0]',state_letters[:,:,0].shape)
# Goal is to have (B,6,5) * 5. a (B,6,5) matrix for each letter for each word batch.
# given a letter (1) check the (6,5) matrix for that letter, mark those places as 1, else 0

def compute_one(state_letters,action_letter,results):
    # takes one word and corresponding state. returns 5,6,5 one hot matrix
    # state_letters (6,5)
    # action_letter (5)
    # results (6,5)
    res = []
    for i,letter in enumerate(action_letter):
        one_hot = torch.zeros(6,5)
        mask=torch.where(state_letters == letter)
        one_hot[mask] = 1
        res.append(one_hot*results)
    return torch.stack(res)

def compute_batch(state_letters,action_letters,result_batch,batch_len):
    attention = [compute_one(state_letters[i],action_letters[i],result_batch[i]) for i in range(batch_len)]
    return torch.stack(attention)

one_hot = compute_one(state_letters[0],action_letters[0],result_batch[0])
print(one_hot.shape)
result_attention = compute_batch(state_letters,action_letters,result_batch,B) # (B,5,6,5)
print('result_attention',result_attention.shape)
state_batch # (B,6,5,3)
# print('one',one_hot)
# # batch example
# one_hot = torch.zeros(B,6,5)
# letter_ex = torch.repeat_interleave(state_letters,5,dim=0).view(5,B,6,5)
# print('letter_ex',letter_ex.shape)
# letter = action_letters[0]
# print(letter,letter_ex)
# mask=torch.where(letter_ex == letter)
# print(mask)
# one_hot[mask] = 1
# print(one_hot)


# one_hot = torch.where(state_letters[:,:,0] == action_words)
# print(one_hot)