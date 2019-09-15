==================================
Deep Deterministic Policy Gradient
==================================

.. contents:: Table of Contents

Background
==========

(Previously: `Introduction to RL Part 1: The Optimal Q-Function and the Optimal Action`_)

.. _`Introduction to RL Part 1: The Optimal Q-Function and the Optimal Action`: ../spinningup/rl_intro.html#the-optimal-q-function-and-the-optimal-action

Deep Deterministic Policy Gradient (DDPG) is an algorithm which concurrently learns a Q-function and a policy. It uses off-policy data and the Bellman equation to learn the Q-function, and uses the Q-function to learn the policy.

This approach is closely connected to Q-learning, and is motivated the same way: if you know the optimal action-value function :math:`Q^*(s,a)`, then in any given state, the optimal action :math:`a^*(s)` can be found by solving