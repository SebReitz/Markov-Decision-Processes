"""
@author: sebastianreitz
"""

'''First MDP'''

import numpy as np
import pandas as pd
import networkx.drawing.nx_pydot as gl
import networkx as nx
import matplotlib.pyplot as plt
from pprint import pprint
import numpy as np
import pandas as pd
import timeit


class Rating(object):
    
    def __init__(self,rating, iteration, simulations):# name, :
        #self.name = name
        self.rating = rating
        self.iteration = iteration
        self.simulations = simulations
            
        # Creating the Probability Transition Matrixs
    tpm = []
    letters = ['AAA','AA','A','BBB','BB','B','CCC','D']
    tpm = pd.DataFrame(tpm,index=letters, columns=letters)

    
    def TPM(self):  
            # Creating the Probability Transition Matrixs
            tpm = np.array(
                    [[1.0,0.0,0,0],[0.3,0.5,0.2,0],[0,0.2,0.5,0.3],[0,0,0.0,1]])
            
            letters = ['A','B','C','D']
            tpm = pd.DataFrame(tpm,index=letters, columns=letters)
            mproduct = tpm
            mi=mproduct
            for _ in range(self.iteration):      
                mproduct = mi
                mi = mproduct.dot(mproduct)
            return mi

states = ['AAA','AA','A','BBB','BB','B','CCC','D']

tpm = Rating(1,0,1)
tpm = tpm.TPM()
T = np.array(tpm)

T = [[1.0, 0.0, 0.0,0.0,0.0,0.0,0.0,0.0],
     [0.1,0.8,0.1,0.0,0.0,0.0,0.0,0.0],
     [0.0,0.1,0.75,0.15,0.0,0.0,0.0,0.0],
     [0.0,0,0.05,0.9,0.03,0.02,0.0,0.0],
     [0.0,0.0,0.0,0.03,0.9,0.06,0.01,0.0],
     [0.0,0.0,0.0,0.0,0.04,0.87,0.05,0.04],
     [0.0,0.0,0.0,0.0,0.02,0.14,0.52,0.32],
     [0.0,0.0,0.0,0.0,0.0,0.0,0.0,1]]

T_test = np.zeros([8,8])
for i in range(8):
    for t in range(8):
        T_test[i][t] = 0.125


T = np.matrix(T)
T=T.T
tm = T.T
tm_graph = pd.DataFrame(tm,columns=states,index=states)
tm = pd.DataFrame(tm)
T = np.array(T)

t_25= np.linalg.matrix_power(tpm, 25)
t_100 = np.linalg.matrix_power(tpm, 100)



R = np.array([
        [1, 0.1, 0.0, -0.1, -0.2, -0.3, -0.4, -1], # R(s_0) -> s'
        [1, 0.1, 0.0, -0.1, -0.2, -0.3, -0.4, -1],  # R(s_1) -> s'
        [1, 0.1, 0.0, -0.1, -0.2, -0.3, -0.4, -1],  # R(s_2) -> s'
        [1, 0.1, 0.0, -0.1, -0.2, -0.3, -0.4, -1],
        [1, 0.1, 0.0, -0.1, -0.2, -0.3, -0.4, -1],
        [1, 0.1, 0.0, -0.1, -0.2, -0.3, -0.4, -1],
        [1, 0.1, 0.0, -0.1, -0.2, -0.3, -0.4, -1],
        [1, 0.1, 0.0, -0.1, -0.2, -0.3, -0.4, -1]# R(s_3) -> s'
    ])


''' Value Iteration'''
start = timeit.default_timer()
v = np.zeros(T.shape[0])
v_old = v.copy()
gamma = 0.1
delta = 1e-5
delta_t = 1
dif = 1
counter = 0

while delta_t > delta:
    counter += 1
    for s in range(len(T)):
        v[s] = np.sum([
                T[s][sp] * (R[s][sp] + gamma * v_old[sp])
                for sp in range(len(T[s]))
            ])
    delta_t = np.sum(np.abs(v - v_old))
    v_old = v.copy()

stop = timeit.default_timer()
print('Time: ', stop - start)
print('')
print(v)
print('\n')
print(counter)


''' Policy Iteration'''
start = timeit.default_timer()
v = np.zeros(T.shape[0])
v_old = v.copy()
gamma = 0.1
delta_t = 1
dif = 1
counter = 0
policy = np.zeros(T.shape[0])
a=[]
T=T.T
two = T
two = pd.DataFrame(two)
policy_old = 0


while delta_t > 0:
    counter += 1
    for s in range(len(T)):
        sp_initial = two.loc[s,:]
        sp_paths = sp_initial[sp_initial!=0.0]
        sp_index = sp_paths.index.values
        v[s] = np.sum([  T[s][sp] * (R[s][sp] + gamma * v_old[sp]) for sp in sp_index   ])
        evaluation = ([  T[s][sp] * (R[s][sp] + gamma * v_old[sp]) for sp in sp_index    ])
        evaluation= pd.DataFrame(evaluation,index=sp_index)
        policy[s] = max(evaluation.idxmax())
        
        a.append([  T[s][sp] * (R[s][sp] + gamma * v_old[sp]) for sp in range(len(T[s])) ])
    delta_t = np.sum(np.abs(policy - policy_old))
    policy_old = policy.copy()

stop = timeit.default_timer()
print('Time: ', stop - start)
print('')
print(policy)
print('\n')
print(counter)







'''Reinforcement Learning'''
start = timeit.default_timer()
rounds = 50
alpha = 0.5
reward = 0
reward_count = []
states = [0,1,2,3,4,5,6,7]
q_value = np.zeros(len(states))
q_value2 = np.zeros([len(states),len(states)])
path_length = 50
iiii = 0

path_count = []
step_count = []
index_chosen = []
total_count = []
convergence = []
convergence2 = []

q = []
q0 = []
q1 = []
q2 = []
q3 = []
q4 = []
q5 = []
q6 = []
q7 = []

for i in range(rounds):
    ''' Initialise a random state in [0,7] to start 
        Then, the possible transitions are calculated and displayed '''
    initial_state = round(np.random.uniform(-0.5,7.5))
#    print(initial_state)
#    print('\n')
    paths = tm.loc[initial_state]
    paths = pd.DataFrame(paths)
    step_count.append(initial_state)
#    print(paths)
    
    if initial_state == 0:
        reward = 1
        q_value[0] = 1
        q_value2[0][0] = 1
        reward_count.append(1)
        continue
    elif initial_state == 7:
        reward = -1
        q_value[7] = -1
        q_value2[7][7] = -1
        reward_count.append(-1)
        continue
    else:
        for ii in range(path_length):
            possible_paths = paths.loc[~(paths<0.001).all(axis=1)]
            possible_paths2 = pd.DataFrame.cumsum(possible_paths)
#            possible_paths2['index1'] = possible_paths2.index
            index = possible_paths2.index.values
#            print(possible_paths)
#            print(possible_paths2)
            

            comparer = round(np.random.uniform(0,1),2)
            for iii in list(possible_paths2.iloc[:,0]):                    
                    
                    if comparer <= iii:
                        index_chosen = index[iiii]
#                        print('')
#                        print(comparer,iii,index_chosen)
#                        print('')
                        next_one = iiii
                        iiii = 0
#                        path_count.append(index_chosen)
                        break
                        
                    else:
                        iiii += 1
#                        print(iiii)
#                        print(comparer,iii)
            path_count.append(index_chosen) 
            q_value[int(possible_paths.columns.values)] = alpha*q_value[int(possible_paths.columns.values)] + (1-alpha)*gamma*(q_value[index_chosen]-q_value[int(possible_paths.columns.values)])
            q_value2[int(possible_paths.columns.values)][int(possible_paths.columns.values)] = alpha*q_value2[int(possible_paths.columns.values)][int(possible_paths.columns.values)] + (1-alpha)*gamma*(q_value2[index_chosen][index_chosen]-q_value2[int(possible_paths.columns.values)][int(possible_paths.columns.values)])
            q_value2[int(possible_paths.columns.values)][index_chosen] = alpha*q_value2[int(possible_paths.columns.values)][int(possible_paths.columns.values)] + (1-alpha)*gamma*(q_value2[index_chosen][index_chosen]-q_value2[int(possible_paths.columns.values)][int(possible_paths.columns.values)])
#            if int(possible_paths.columns.values) == 4:
#                print(index_chosen)
#            q_value[index_chosen] = q_value[int(possible_paths.columns.values)]
            if index_chosen == 0 or index_chosen == 7:
                if index_chosen == 0:
                    q_value[0] = 1
#                    q_value2[0][index_chosen] = 1
                elif index_chosen == 7:
                    q_value[7] = -1
#                    q_value2[7][index_chosen] = -1
                break

            paths = tm.loc[index_chosen]
            paths = pd.DataFrame(paths)
#            print(index_chosen)
#    alpha = alpha + 0.0002
    total_count.append([step_count,path_count])
#    q.append(q_value)
    convergence.append(sum(sum(q_value2[1:-1][:])))
    convergence2.append(q_value2)
    q0.append(q_value[0])
    q1.append(q_value[1])
    q2.append(q_value[2])
    q3.append(q_value[3])
    q4.append(q_value[4])
    q5.append(q_value[5])
    q6.append(q_value[6])
    q7.append(q_value[7])
#    print(q_value)
            


stop = timeit.default_timer()
print('Time: ', stop - start) 


''' Plot convergence'''
axis = np.linspace(0,len(convergence),len(convergence))
plt.figure()
plt.title('Convergence per Rounds played')
plt.plot(axis,convergence)
plt.xlabel('Rounds')
plt.ylabel("Sum Q's")
    
    
    
    

    
    
    
    
''' Plotting the Markov Chain'''
q_df = tm
q = q_df.values
print('\n')
print(q, q.shape)
print('\n')
print(q_df.sum(axis=1))




def _get_markov_edges(Q):
    edges = {}
    for col in Q.columns:
        for idx in Q.index:
            edges[(idx,col)] = Q.loc[idx,col]
    return edges

edges_wts = _get_markov_edges(q_df)
pprint(edges_wts)

G = nx.MultiDiGraph()

G.add_nodes_from(states)
print('Nodes:\n')
print(G.nodes())
print('\n')

for k, v in edges_wts.items():
    tmp_origin, tmp_destination = k[0], k[1]
    G.add_edge(tmp_origin, tmp_destination, weight=v, label=v)
print('Edges:')
pprint(G.edges(data=True))

plt.figure(figsize=(20,20))

pos = nx.drawing.nx_pydot.graphviz_layout(G, prog='dot')
nx.draw_networkx(G, pos)

edge_labels = {(n1,n2):d['label'] for n1,n2,d in G.edges(data=True)}
nx.draw_networkx_edge_labels(G , pos, edge_labels=edge_labels)
nx.drawing.nx_pydot.write_dot(G, 'markov.dot')
plt.show()



'''Second MDP'''



#from gym.envs.registration import register
#register(
#    id='FrozenLakeNotSlippery-v19',
#    entry_point='frozen_lake16:FrozenLakeEnv',
#    kwargs={'map_name' : '16x16', 'is_slippery': True},
#    max_episode_steps=100,
#    reward_threshold=0.78, # optimum = .8196
#)

#from gym.envs.registration import register
#register(
#    id='FrozenLakeNotSlippery-v18',
#    entry_point='frozen_lake16:FrozenLakeEnv',
#    kwargs={'map_name' : '16x16', 'is_slippery': False},
#    max_episode_steps=100,
#    reward_threshold=0.78, # optimum = .8196
#)

from frozen_lake16 import FrozenLakeEnv
import matplotlib.pyplot as plt
import timeit
import random
import numpy as np
import gym
from gym import wrappers


'''Value Iteration'''
####################



def run_episode(env, policy, gamma , render = False):
    """ Evaluates policy by using it to run an episode and finding its
    total reward.
    args:
    env: gym environment.
    policy: the policy to be used.
    gamma: discount factor.
    render: boolean to turn rendering on/off.
    returns:
    total reward: real value of the total reward recieved by agent under policy.
    """
    obs = env.reset()
    total_reward = 0
    step_idx = 0
    while True:
        if render:
            env.render()
        obs, reward, done , _ = env.step(int(policy[obs]))
        total_reward += (gamma ** step_idx * reward)
        step_idx += 1
        if done:
            break
    return total_reward


def evaluate_policy(env, policy, gamma,  n = 100):
    """ Evaluates a policy by running it n times.
    returns:
    average total reward
    """
    scores = [
            run_episode(env, policy, gamma = gamma, render = False)
            for _ in range(n)]
    return np.mean(scores)

def extract_policy(v, gamma):
    """ Extract the policy given a value-function """
    policy = np.zeros(env.nS)
    for s in range(env.nS):
        q_sa = np.zeros(env.action_space.n)
        for a in range(env.action_space.n):
            for next_sr in env.P[s][a]:
                # next_sr is a tuple of (probability, next state, reward, done)
                p, s_, r, _ = next_sr
                q_sa[a] += (p * (r + gamma * v[s_]))
        policy[s] = np.argmax(q_sa)
    return policy


def value_iteration(env, gamma ):
    """ Value-iteration algorithm """
    v = np.zeros(env.nS)  # initialize value-function
    tracker = []
    max_iterations = 100000
    eps = 0.001
    for i in range(max_iterations):
        prev_v = np.copy(v)
        for s in range(env.nS):
            q_sa = [sum([p*(r + prev_v[s_]) for p, s_, r, _ in env.P[s][a]]) for a in range(env.nA)] 
            v[s] = max(q_sa)
        tracker.append(np.sum(np.fabs(prev_v - v)))
        if (np.sum(np.fabs(prev_v - v)) <= eps):
            print ('Value-iteration converged at iteration# %d.' %(i+1))
            break
    return v,tracker


if __name__ == '__main__':
    '''Deterministic'''
    start = timeit.default_timer()
    env_name  = 'FrozenLakeNotSlippery-v18'
    gamma = 0.9
    env = gym.make(env_name)
    env = env.unwrapped
    optimal_v,tracker = value_iteration(env, gamma)  
    policy = extract_policy(optimal_v, gamma)
    stop = timeit.default_timer()
    print('Time: ', stop - start)



    '''Probabilistic'''
    start = timeit.default_timer()
    env_name2  = 'FrozenLakeNotSlippery-v19'
    gamma = 0.9
    env = gym.make(env_name2)
    env = env.unwrapped
    optimal_v2,tracker2 = value_iteration(env, gamma) 
    policy2 = extract_policy(optimal_v2, gamma)  
    stop = timeit.default_timer()
    print('Time: ', stop - start)
    
    
    plt.figure()
    plt.plot(np.linspace(0,len(tracker),len(tracker)),tracker)
    plt.title('Deterministic Case: Difference in Values by Number of Iterations')
    plt.xlabel('Iterations')
    plt.ylabel('Delta')
    plt.show()
    
    plt.figure()
    plt.plot(np.linspace(0,len(tracker2),len(tracker2)),tracker2)
    plt.title('Probabilistic Case: Difference in Values by Number of Iterations')
    plt.xlabel('Iterations')
    plt.ylabel('Delta')
    plt.show()
    
'''Policy Iteration'''
#####################



def run_episode(env, policy, gamma, render = False):
    """ Runs an episode and return the total reward """
    obs = env.reset()
    total_reward = 0
    step_idx = 0
    while True:
        if render:
            env.render()
        obs, reward, done , _ = env.step(int(policy[obs]))
        total_reward += (gamma ** step_idx * reward)
        step_idx += 1
        if done:
            break
    return total_reward


def evaluate_policy(env, policy, gamma , n = 100):
    scores = [run_episode(env, policy, gamma, False) for _ in range(n)]
    return np.mean(scores)

def extract_policy(v, gamma = 1.0):
    """ Extract the policy given a value-function """
    policy = np.zeros(env.nS)
    for s in range(env.nS):
        q_sa = np.zeros(env.nA)
        for a in range(env.nA):
            q_sa[a] = sum([p * (r + gamma * v[s_]) for p, s_, r, _ in  env.P[s][a]])
        policy[s] = np.argmax(q_sa)
    return policy

def compute_policy_v(env, policy, gamma):
    """ Iteratively evaluate the value-function under policy.
    Alternatively, we could formulate a set of linear equations in iterms of v[s] 
    and solve them to find the value function.
    """
    v = np.zeros(env.nS)
    eps = 0.001
    while True:
        prev_v = np.copy(v)
        for s in range(env.nS):
            policy_a = policy[s]
            v[s] = sum([p * (r + gamma * prev_v[s_]) for p, s_, r, _ in env.P[s][policy_a]])
        if (np.sum((np.fabs(prev_v - v))) <= eps):
            # value converged
            break
    return v

def policy_iteration(env, gamma ):
    """ Policy-Iteration algorithm """
    np.random.seed(0)
    policy = np.random.choice(env.nA, size=(env.nS))  
    max_iterations = 200000
    gamma = 0.9
    
    for i in range(max_iterations):
        old_policy_v = compute_policy_v(env, policy, gamma)
        new_policy = extract_policy(old_policy_v, gamma)
        if (np.all(policy == new_policy)):
            print ('Policy-Iteration converged at step %d.' %(i+1))
            break
        policy = new_policy
    return policy


if __name__ == '__main__':
    '''Deterministic'''
    start = timeit.default_timer() 
    env_name  = 'FrozenLakeNotSlippery-v18'
    gamma = 0.9
    env = gym.make(env_name)
    env = env.unwrapped
    optimal_policy = policy_iteration(env, gamma)
    stop = timeit.default_timer()
    print('Time: ', stop - start)

    
    '''Probabilistic'''
    start = timeit.default_timer()
    env_name2  = 'FrozenLakeNotSlippery-v19'
    gamma = 0.9
    env2 = gym.make(env_name2)
    env2 = env2.unwrapped
    optimal_policy2 = policy_iteration(env2, gamma)
    stop = timeit.default_timer()
    print('Time: ', stop - start)





'''Q-Learning'''
env = gym.make('FrozenLakeNotSlippery-v19')
env = env.unwrapped
env.seed(0)
np.random.seed(0)
q_learning_table = np.zeros([env.observation_space.n,env.action_space.n])

# -- hyper --
num_epis = 8500
num_iter = 850
learning_rate = 0.5
discount = 0.9

# -- training the agent ----
for epis in range(num_epis):
    state = env.reset()
    for iter in range(num_iter):
        action = np.argmax(q_learning_table[state,:] + np.random.randn(1,4))
        state_new,reward,done,_ = env.step(action)
        q_learning_table[state,action] = (1-learning_rate)* q_learning_table[state,action] + \
                                         learning_rate * (reward + discount*np.max(q_learning_table[state_new,:]) )
        state = state_new
        if done: break
print(np.argmax(q_learning_table,axis=1))
print(np.around(q_learning_table,6))
print('-------------------------------')

# visualize no uncertainty
tracker = []
s = env.reset()
for _ in range(1000):
    action  = np.argmax(q_learning_table[s,:])
    tracker.append(action)
    state_new,_,done,_ = env.step(action)
    env.render()
    s = state_new
    if done: break
print('-------------------------------')









    
    
    
    
    

