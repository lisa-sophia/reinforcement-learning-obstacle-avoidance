#!/usr/bin/env python3

import matplotlib.pyplot as plt
import pickle
import rospkg
import numpy as np

# load the Q-table
rospack = rospkg.RosPack()
pkg_path = rospack.get_path('reinforcement_learning')
file_dir = pkg_path + '/training_results/final_q_values.pkl'
with open(file_dir, 'rb') as f:
    q_table = pickle.load(f)

grid = []
max_q = []
for key in q_table:
    s, a = key
    q_values = []
    i = 0
    while i < 4:
        try:
            q_values.append(q_table[(s, i)])
            i += 1
        except:
            i += 1
    max_q.append(max(q_values))
    if s[0] == "-":
        x_pos = s[0:4]
        if s[4] == "-":
            y_pos = s[4:8]
        else:
            y_pos = s[4:7]
    else:
        x_pos = s[0:3]
        if s[3] == "-":
            y_pos = s[3:7]
        else:
            y_pos = s[3:6]
    grid.append([float(x_pos), float(y_pos)])

# calculate the "average" max q-value for each x-y-position 
# (averaged over the different pedestrian positions)
unique_grid = [list(x) for x in set(tuple(x) for x in grid)]
unique_q_max = []
for pos in unique_grid:
    max_q_values = []
    for i in range(len(grid)):
        if pos[0] == grid[i][0] and pos[1] == grid[i][1]:
            max_q_values.append(max_q[i])
    unique_q_max.append(sum(max_q_values)/len(max_q_values))
    #unique_q_max.append(max(max_q_values))

x_min = min([unique_grid[i][0] for i in range(len(unique_grid))])
y_min = min([unique_grid[i][1] for i in range(len(unique_grid))])
x_max = max([unique_grid[i][0] for i in range(len(unique_grid))])
y_max = max([unique_grid[i][1] for i in range(len(unique_grid))])

print(x_min, x_max)
print(y_min, y_max)

merged_list = []
for i in range(len(unique_grid)):
    merged_list.append([unique_grid[i], unique_q_max[i]])

#merged_list.sort(key = lambda x: x[0][1]) # sort by y-coordinate

def get_values(iterables, key_to_find):
    for keys in iterables:
        x = keys[0][0]
        y = keys[0][1]
        if x == key_to_find[0] and y == key_to_find[1]:
            return keys[1]
    return None

# matrix with x times y values:
q_matrix = np.zeros( (int((x_max - x_min)/0.5), int((y_max - y_min)/0.5)) )
# 0,0: x=6.0, y=1.5  ;  1,0: x=6.0, y=1.0
# 0,1: x=5.5, y=1.5  ;  1,1: x=5.5, y=1.0
# 0,2: x=5.0, y=1.5  ;  1,2: x=5.0, y=1.0
# ...

x_labels = []
y_labels = []
min_matrix_value = min(unique_q_max) - 0.5
for i in range(np.shape(q_matrix)[0]):
    for j in range(np.shape(q_matrix)[1]):
        x = x_max - 0.5*i
        y = y_max - 0.5*j
        q_val = get_values(merged_list, [x, y])
        if q_val is None:
            q_matrix[i,j] = min_matrix_value
        else:
            q_matrix[i,j] = q_val
        if i == 0:
            y_labels.append(str(y))
    x_labels.append(str(x))

fig, ax = plt.subplots()
im = ax.imshow(q_matrix)

# show all ticks with respective labels (coordinates)
# x and y are swapped, to resemble ROS axes
ax.set_xticks(np.arange(np.shape(q_matrix)[1]))
ax.set_yticks(np.arange(np.shape(q_matrix)[0]))
ax.set_xticklabels(y_labels)
ax.set_yticklabels(x_labels)

ax.set_title("Average max q-value per state")
fig.colorbar(im)
fig.tight_layout()
plt.show()


## plot the rewards
file_dir = pkg_path + '/training_results/rewards.txt'
with open(file_dir, 'r') as f:
    rewards = []
    for line in f.readlines():
        rewards.append(float(line))

x = np.arange(len(rewards))

plt.subplot(2, 1, 1)
plt.plot(x, rewards)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Rewards per epsiode')

# average reward over nr. of episodes:
sample_size = 25
samples = int(len(rewards)/sample_size)
episodes = []
averaged = []

for i in range(samples):
    rewards_split = rewards[i*sample_size:(i+1)*sample_size]
    averaged.append(sum(rewards_split)/len(rewards_split))
    episodes.append((i+1)*sample_size)

plt.subplot(2, 1, 2)
plt.plot(episodes, averaged)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Average rewards per ' + str(sample_size) + ' epsiodes')

plt.tight_layout()
plt.show()