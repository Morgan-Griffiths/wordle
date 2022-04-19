import numpy as np
import math
import random

"""
Priority Tree.
3 tiered tree structure containing
Root node (Object. sum of all lower values)
Intermediate Node (Object. Root as parent, sums a given slice of the priority array)
Priority Array (Array of priorities, length buffer_size)
The number of Intermediate nodes is calculated by the buffer_size / batch_size.
I_episode: current episode of training
Index: is calculated by i_episode % buffer_size. This loops the index after exceeding the buffer_size.
Indicies: (List) of memory/priority entries
intermediate_dict: maps index to intermediate node. Since each Intermediate node is responsible 
for a given slice of the priority array, given a particular index, it will return the Intermediate node
'responsible' for that index.
## Functions:
Add:
Calculates the priority of each TD error -> (abs(TD_error)+epsilon)**alpha
Stores the priority in the Priority_array.
Updates the sum_tree with the new priority
Update_Priorities:
Updates the index with the latest priority of that sample. As priorities can change over training
for a particular experience
Sample:
Splits the current priority_array based on the number of entries, by the batch_size.
Returns the indicies of those samples and the priorities.
Propogate:
Propogates the new priority value up through the tree
"""


class PriorityTree(object):
    def __init__(self, buffer_size, batch_size, alpha, epsilon):
        self.alpha = alpha
        self.epsilon = epsilon
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.batch_indicies = np.arange(0, self.batch_size)

        self.num_intermediate_nodes = math.ceil(buffer_size / batch_size)
        self.current_intermediate_node = 0
        self.root = Node(None)
        self.intermediate_nodes = [
            Intermediate(self.root, batch_size * x, batch_size * (x + 1))
            for x in range(self.num_intermediate_nodes)
        ]
        self.priority_array = np.zeros(buffer_size)
        self.intermediate_dict = {}
        for index, node in enumerate(self.intermediate_nodes):
            for key in range(
                (batch_size * (index + 1)) - batch_size, batch_size * (index + 1)
            ):
                self.intermediate_dict[key] = node
        print(
            "Priority Tree: Batch Size {} Buffer size {} Number of intermediate Nodes {}".format(
                batch_size, buffer_size, self.num_intermediate_nodes
            )
        )

    def add(self, TD_error, index):
        priority = (abs(TD_error) + self.epsilon) ** self.alpha
        self.priority_array[index] = priority
        # Update sum
        propogate(self.intermediate_dict[index], self.priority_array)

    def sample(self, index, limit):
        spacing = np.linspace(0, limit - self.batch_size, self.batch_size, dtype=np.int)
        random_indicies = np.random.choice(self.batch_indicies, size=self.batch_size)
        indicies = random_indicies + spacing
        priorities = self.priority_array[indicies]
        return priorities, indicies

    def update_priorities(self, TD_errors, indicies):
        priorities = (np.abs(TD_errors) + self.epsilon) ** self.alpha
        self.priority_array[indicies] = priorities
        # Update sum
        nodes = [self.intermediate_dict[index] for index in indicies]
        intermediate_nodes = set(nodes)
        [propogate(node, self.priority_array) for node in intermediate_nodes]


class Node(object):
    def __init__(self, parent):
        self.parent = parent
        self.children = []
        self.value = 0

    def add_child(self, child):
        self.children.append(child)

    def set_value(self, value):
        self.value = value

    def sum_children(self):
        return sum([child.value for child in self.children])

    def __len__(self):
        return len(self.children)


class Intermediate(Node):
    def __init__(self, parent, start, end):
        self.parent = parent
        self.start = start
        self.end = end
        self.value = 0
        parent.add_child(self)

    def sum_leafs(self, arr):
        return np.sum(arr[self.start : self.end])


def propogate(node, arr):
    if node.parent != None:
        node.value = node.sum_leafs(arr)
        propogate(node.parent, arr)
    else:
        node.value = node.sum_children()
