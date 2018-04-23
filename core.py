import random
import torch
import numpy as np
import utils
from policy import GreedyEpsilonPolicy
from torch.autograd import Variable

class SegmentTree():
  def __init__(self, size, initial_max=1):
    self.oldest_idx = 0
    self.size = size
    self.full = False  # Used to track actual capacity
    self.sum_tree = [0] * (2 * size - 1)  # Initialise fixed size tree with all (priority) zeros
    self.max_tree = [initial_max] * (2 * size - 1)  # Initialise with priorities (incorrect as max will always be >= 0, but sum tree is used for sampling)
    self.data = [Sample(np.zeros([84,84], dtype = np.uint8), None, 0 , np.zeros([84,84],dtype = np.uint8), False, -1)]*size
    
  def _propagate(self, index, value):
    parent = (index - 1) // 2
    left, right = 2 * parent + 1, 2 * parent + 2
    self.sum_tree[parent] = self.sum_tree[left] + self.sum_tree[right]
    self.max_tree[parent] = max(self.max_tree[left], self.max_tree[right])
    if parent != 0:
      self._propagate(parent, value)

  # Updates value given a tree index
  def update(self, index, value):
    self.sum_tree[index], self.max_tree[index] = value, value  # Set new value
    self._propagate(index, value)  # Propagate value
  def append(self, sample, value):
    self.data[self.oldest_idx] = sample ## can use copy.deepcopy
    self.update(self.oldest_idx + self.size - 1, value)
    self.oldest_idx = (self.oldest_idx + 1)%self.size
    self.full = self.full or self.oldest_idx==0
    
  ###############################
  # Searches for the location of a value in sum tree
  def _retrieve(self, index, value):
    left, right = 2 * index + 1, 2 * index + 2
    if left >= len(self.sum_tree):
      return index
    elif value <= self.sum_tree[left]:
      return self._retrieve(left, value)
    else:
      return self._retrieve(right, value - self.sum_tree[left])

  # Searches for a value in sum tree and returns value, data index and tree index
  def find(self, value):
    index = self._retrieve(0, value)  # Search for index of item from root
    data_index = index - self.size + 1
    return (self.sum_tree[index], data_index, index)  # Return value, data index, tree index

  # Returns data given a data index
  def get(self, data_index):
    return self.data[data_index % self.size]

  def total(self):
    return self.sum_tree[0]

  def max(self):
    return self.max_tree[0]
###################################################################


###################################################################
## Incorporating PER : Need to change the definition of Sample 
#Adding 'timestep' attribute to the Sample object

class Sample(object):
    def __init__(self, state, action, reward, next_state, end, timestep):
        utils.assert_eq(type(state), type(next_state))
        #print 'State ' + str(type(state))
        self._state = (state * 255.0).astype(np.uint8)
        self._next_state = (next_state * 255.0).astype(np.uint8)
        self.action = action
        self.reward = reward
        self.end = end
	    self._timestep = timestep
    @property
    def state(self):
        return self._state.astype(np.float32) / 255.0
   
    def timestep(self):
	return self._timestep

    @property
    def next_state(self):
        return self._next_state.astype(np.float32) / 255.0

    def __repr__(self):
        info = ('S(mean): %3.4f, A: %s, R: %s, NS(mean): %3.4f, End: %s'
                % (self.state.mean(), self.action, self.reward,
                   self.next_state.mean(), self.end))
        return info
#################################################################
## Incorporating PER : In order to incorporate PER, 1)we need to add 'priority weight' in the arguments
# 2) We also need to add 'priority exponent'
##################################################################
class ReplayMemory(object):
    def __init__(self, args):
	    self.dtype_byte = torch.cuda.ByteTensor #if args.cuda else torch.ByteTensor
	    self.dtype_long = torch.cuda.LongTensor #if args.cuda else torch.LongTensor
	    self.dtype_float = torch.cuda.FloatTensor #if args.cuda else torch.FloatTensor

        self.max_size = args.replay_buffer_size
	    self.priority_exponent = args.priority_exponent
	    self.priority_weight = args.priority_weight
        self.transitions = SegmentTree(self.max_size, 1**self.priority_weight)
	    self.t = 0 

    def __len__(self):
        return len(self.samples)

    def _evict(self):
        """Simplest FIFO eviction scheme."""
        to_evict = self.oldest_idx
        self.oldest_idx = (self.oldest_idx + 1) % self.max_size
        return to_evict

    def burn_in(self, env, agent, num_steps):
	    policy = GreedyEpsilonPolicy(1,agent)
	    i = 0
    	while i< num_steps or not env.end:
		    if env.end:
			    state = env.reset()
		    action = policy.get_action(None)
		    next_state, reward = env.step(action)
		    self.append(state, action, reward, next_state, env.end)
		    state = next_state
		    i+=1
		    if i%10000 == 0:
			    print '%d framed burned in' %i
	    print '%d frames burned into memory.' %i

    def append(self, state,action, reward, next_state, end):
	    assert len(self.transitions.data)<=self.max_size
   	    new_sample = Sample(state, action, reward, next_state,end,self.t)
 	    self.transitions.append(new_sample, self.transitions.max())
	    self.t+=1

    def _get_sample_from_segment(self, segment, i):
        sample = random.uniform(i*segment, (i+1)*segment)
	    prob, idx, tree_idx = self.transitions.find(sample)
	    return prob, idx, tree_idx, self.transitions.get(idx)	
   
    def sample(self, batch_size):
	    p_total = self.transitions.total()
	    segment = p_total/batch_size
	    batch = [self._get_sample_from_segment(segment, i) for i in range(batch_size)]
	    probs, idxs, tree_idxs, batch_samples = zip(*batch)
	    probs = Variable(self.dtype_float(probs)) / p_total
	    capacity = self.max_size if self.transitions.full else self.transitions.oldest_idx
	    weights = (capacity*probs)** -self.priority_weight
	    weights = weights/weights.max()
	    return tree_idxs, batch_samples, weights 
	
	
    ##################################################
    
    def update_priorities(self, idxs, priorities):
	    priorities.pow_(self.priority_exponent)
	    [self.transitions.update(idx, priority) for idx, priority in zip(idxs, priorities)]


    def clear(self):
        self.samples = []
        self.oldest_idx = 0


def samples_to_tensors(samples):
    num_samples = len(samples)

    states_shape = (num_samples, ) + samples[0].state.shape
    states = np.zeros(states_shape, dtype=np.float32)
    next_states = np.zeros(states_shape, dtype=np.float32)

    rewards = np.zeros(num_samples, dtype=np.float32)
    actions = np.zeros(num_samples, dtype=np.int64)
    non_ends = np.zeros(num_samples, dtype=np.float32)

    for i, s in enumerate(samples):
        states[i] = s.state
        next_states[i] = s.next_state
        rewards[i] = s.reward
        actions[i] = s.action
        non_ends[i] = 0.0 if s.end else 1.0

    states = torch.from_numpy(states).cuda()
    actions = torch.from_numpy(actions).cuda()
    rewards = torch.from_numpy(rewards).cuda()
    next_states = torch.from_numpy(next_states).cuda()
    non_ends = torch.from_numpy(non_ends).cuda()

    return states, actions, rewards, next_states, non_ends
