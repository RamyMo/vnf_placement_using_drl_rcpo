# Optimizing VNF Placement Using DRL and RCPO


This repository contains a code implementation for optimizing Virtual Network Function Placement using Deep Reinforcement Learning (DRL) and Reward Constrained Policy Optimization (RCPO).

This project utilizes an attentional sequence-to-sequence model to predict real-time solutions in a highly constrained environment. To achieve this, additional reward signals are used to estimate the agent's parameters. The reward signal, which indicates the VNF placememnt cost, is enhanced with feedback signals that reflect the degree of constraint violations. These constraints are incorporated into the cost function through the Lagrange relaxation technique. Furthermore, the Lagrange multipliers are updated automatically using Reward Constrained Policy Optimization (RCPO).

# Repository Structure

- **agent.py:** Contains the implementation of the DRL agent.
- **config.py:** Configuration file for setting up parameters.
- **environment.py:** Defines the environment for VNF placement.
- **service_batch_generator.py:** Generates batches of service requests.
- **main.py:** Main script to execute the training and evaluation of the model.
  
# Install Requirements 

````
    conda create --name env_name --file requirements.txt
````
# Model Training

```
python main.py --learn_mode=True --save_model=True --save_to=save/l12 --num_epoch=20000 --min_length=12 --max_length=12 --num_layers=1 --hidden_dim=32 --num_cpus=10 --env_profile="small_default"
```
Train a model for 20,000 epochs on SFCs of length 12 using a single-layer neural network with 32 hidden units, simulating the environment with 10 CPUs and saving the trained model to the specified directory.




# Debug

To visualize training variables on Tensorboard:
```
    tensorboard --logdir=summary/repo
```

# Learning History

To plot the learning history run:
```
    python graphicate_learning.py -f save/model/learning_history.csv
```

# Paper
Mohamed, Ramy, et al. ”Optimizing Resource Fragmentation in Virtual Network
Function Placement using Deep Reinforcement Learning”, Under Review,
IEEE Transactions on Machine Learning in Communications and Networking.
