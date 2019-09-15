==================================
Designing the API
==================================

The API will decide how various modules of your code will talk to each other.
Any reinforcement library consists of 4 main submodules -

1. *Environment* -This module has various environments with openAI gym compatible API.
The environments are contained in core/environments/<environment_name> folder

2. *Agents* - Various RL algorithms primarily differ in the way agent records/uses/learns from state/reward information.
Every algorithm has a separate agent class with a common API. The agents are contained in core/agents/<algorithm_name>

3. *Policies/Value/Q functions* - This library only features neural network based policies.
Policies differ in the way the accept input and what output they produce but share a common API.
Different agents may use the same policies. Policies include NNs that approximate Q(s,a), V(s), pi(s) and hybrids or those
The policies are contained in core/policies/<algorithm_name>

4. *Utilities* - Utilities includes pre-processing, memory replay buffers, frame stacking buffers etc which are shared across algorithms.
They are contained in core/utils

NOTE: Since different agents/environments/policies share same API, other libraries group them into abstract base classes.
Such technicalities are left out for ease of understanding.

We will follow a top down approach. We will first write what a standard user would want to code to use any RL algorithm.
Lets have a look at training file for DQN (algorithm does not matter, we are only going to understand the structure)