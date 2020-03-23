# Financial Application - Optimal Market Making

This is an implementation of the Optimal Market Making reinforcement learning problem posed in CME 241 Winter 2020. 

## Problem Statement

Over a series of T time periods, we would like to buy and sell shares as a market maker with the goal of maximizing
our final expected utility of wealth. This wealth includes all intermediate proceeds as well as final proceeds from
closing out any existing inventory.

## RL Algorithm Structure

This implementation uses tabular Q-Learning and SARSA (from the 'rl' folder) to solve the problem. States are a tuple of time, stock price, wealth, and existing inventory. Actions are the amount and price to place bids and asks. Transitions are governed by a random process of bids being hit or asks being lifted (currently via a number of coin flip Bernoulli's dependent on the size of the spread offered). The only reward is the reward for terminal wealth and the closing of existing inventory at a penalty factor.

## Images

Two images are provided, one for the results of Q-Learning and one for Sarsa, showing decisions at the pre-terminal timestep. Results are very similar between the two algorithms for a basic run of state size ~100,000, action size 36, and 10,000 iterations. The algorithm currently favors selling shares across situations, likely because the existing inventory penalty could use further calibration. However, actions do seem to vary across the possible options and the algorithm does favor buying shares to clear inventory in some circumstances.

## Shortcomings and Potential Improvements

The major existing shortcoming is the tabular implementation. For reasonable computational time, the state size and action size must be severely limited. This poses a particular problem in this setting due to the ranges of the variables. Our wealth range must be wide if we want flexibility in the number of shares bought/sold, however precision is also necessary in order to capture the difference between selling or buying at different spread levels. Additionally, the tabular nature does not allow for easy understanding of the algorithms decision making. The images provided only show decisions against a singular dimension. Value function approximation, particularly linear, may provide more interpretable results by directly showing the relationship between states/actions and value. 

Another potential major improvement would be to further specify the simulator for state transitions. Currently, the market dynamics are simulated very rudimentarily (by random Bernoulli draws) and returns drawn from a normal distribution (note: must be rounded to fit tabular implementation). Having other agents in the simulator, modeling the TOB, or simply having more realistic probability distributions of transitions would greatly improve the realism of the model. 