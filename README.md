# Invariant Measures in Delay-Coordinates

This repository contains python code which can be used to reproduce the numerical experiments in our work *Invariant Measures in Time-Delay Coordinates for Unique Dynamical System Idenfitification.* 

`torus_rotation.py`: This file simulates trajectories of four different torus rotation maps and visualizes the corresponding state-coordinate invariant measures and delay-coordinate invariant measures.

`NN_measure_loss.py`: This file trains a neural network to learn the dynamics of the Lorenz-63 system using loss functions based on either the state-coordinate or delay-coordinate invariant meausre.

`plot_training_result.py`: This file plots the result of the neural network training. 

Running the `torus_rotation.py` file will create the following graphic, illustrating the invariant measures of the torus rotation for different parameter values in both the state-coordinate and delay-coordinate axes. 

![image](https://github.com/user-attachments/assets/fd010190-c71f-44cf-b31d-a6c9890b1220)


Running the `NN_measure_loss.py` file and then `plot_training_result.py` will create the following graphic, which shows the training result when a standard MLP is used to learn the flow map based on either a state-coordinate or delay-coordinate invariant measure loss. 

