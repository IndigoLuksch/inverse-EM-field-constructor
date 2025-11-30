# Inverse EM field constructor

Goal: 
- construct an arrangement of magnets that produces a desired magnetic field

Approach: 
1. train a neural network to predict properties of a single magnet given the magnetic field it produces
2. input a desired magnetic field into the model
3. subtract the magnetic field produced by the magnet it outputs
4. repeat this cycle 

## Files 

[E_field_notebook](1_initial_attempt/E_field_notebook.jpynb) notebook contains initial attempts create a convolutional NN for an electric field
