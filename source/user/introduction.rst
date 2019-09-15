============
Introduction
============

.. contents:: Table of Contents

What This Is
============

Welcome to Deep RL with CNTK

This repository is meant to teach the intricacies of writing advanced Deep Reinforcement Algorithms in CTNK.
In short, a lot of good repositories for RL exists. Most of them are easy to USE libraries rather than easy to UNDERSTAND libraries.
There is a steep learning curve for someone new to the field who wants to modify existing architectures or explore possibilities.


Why I Built This
=================

I observed that a lot of labs around universities were trying to write custom implementations of popular RL algorithms
such that they suit the needs of their environments. Using existing libraries proved challenging either due to non standard environments,
inability to tweak the library code, or just the complexity of implementation. The issue is exacerbated by the fact that debugging RL is not the same as debugging a traditional software code.
An incorrect implementation wont throw an error. It will often manifest itself late in the rewards graph during training. By then, the researcher has already spent days training the algorithm.
Using existing libraries as a guide to write your own algorithms is also challenging due to the complexities in their implementations. All of this is a huge barrier for someone new to the field of RL.

To overcome that, I wrote this library which guides you on how to write your own library. How will this be better -
- Each algorithm is self contained avoiding cross dependencies
- The common code is consistent across algorithms
- All algorithms are implemented in single agent, single environment version. No confusing parallelisation or extra dimension of agents/environments
- No tricks. All algorithms are implemented in the most standard way described in reference papers
- Lots, lots and lots of comments. Each code line commented. Way more comments than needed
