CPSC 532J: Never-Ending RL\
Fall Term 2021\
Assignment 2\
University of British Columbia\
Ryan Fayyazi


### Part 1

Instructions: Implement Policy Gradients for a neural network doing something.

* Task: OpenAI Gym CartPole-v0
* Algorithm: Episodic One-step Actor-Critic implementation based on [S&B20] page 332

Run with `$ python AC.py`

default arguments:

```bash
--N 7500 --gamma 0.97 --actor_lr 0.001 --critic_lr 0.001 --actor_dims 64 64 --critic_dims 64 64 --track_param False --track_logp True --render False --render_step 1000
```
### Part 2

Instructions: Implement Genetic Algorithm for a neural network doing something.

* Task: OpenAI Gym CartPole-v0
* Algorithm: Genetic Algorithm implementation based on [Suc17]

Run with `$ python GA.py`

default arguments:

```bash
--G 1000 --N 100 --T 20 --n_candidates 10 --sigma 0.005 --hidden_dims 64 64 --track_param False
```

### Part 3 

Instructions: Do a bit of research (something new)

#### Experiment 1

This experiment investigates a hybrid approach between policy gradients and genetic algorithms. The approach is to perform 
stochastic gradient descent on each policy in the population after each mutation step, via the REINFORCE algorithm ([S&B20]
page 328) for a single episode. 

Run with `$ python GAPG.py`

default arguments:

```bash
--G 1000 --N 1000 --T 20 --n_candidates 10 --sigma 0.005 --gamma 0.97 --lr 0.001 --hidden_dims 64 64 --track_param False
```

### Bonus

* Task: OpenAI Gym CartPole-v0
* Algorithm: Evolutionary Strategies implementation based on [Sal17]

Run with `$ python ES.py`

default arguments:

```bash
--G 1000 --N 100 --lr 0.001 --sigma 0.1 --hidden_dims 64 64 --track_param False
```

### References

[S&B20] - Richard S. Sutton & Andrew G. Barto. Reinforcement Learning: An Introduction (Second Edition). 2020.

[Suc17] - Felipe Petroski Such, Vashisht Madhavan, Edoardo Conti, Joel Lehman, Kenneth O. Stanley, & Jeff Clune. Deep Neuroevolution: Genetic Algorithms are a Competitive Alternative for
Training Deep Neural Networks for Reinforcement Learning. https://arxiv.org/pdf/1712.06567.pdf. 2017.

[Sal17] - Tim Salimans, Jonathan Ho, Xi Chen, Szymon Sidor & Ilya Sutskever. Evolutionary Strategies as a Scalable \
Alternative to Reinforcement Learning. https://arxiv.org/pdf/1703.03864.pdf. 2017. 