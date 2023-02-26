import numpy as np

from neural_network import NeuralNetwork
from neural_network import ActivationFunction

from window import Window

import math

def rule_topleft(x, y, hdg, gen):
    """
    In the first 50 generations, any creature ending up at the right half of 
    the field survives. Between 50 and 100 generations the border of survival 
    is gradually shifted to the right. From generation 100 any creature 
    ending up at the right quarter of the field survives.
    """
    if gen < 50:
        num = 0
    elif gen >= 50 and gen < 100:
        num = (gen - 50) / 100
    else:
        num = 0.5
    
    return x < num and y > num

def rule_topright(x, y, hdg, gen):
    if gen < 50:
        num = 0
    elif gen >= 50 and gen < 100:
        num = (gen - 50) / 100
    else:
        num = 0.5
    
    return x > num and y > num

def rule_bottomleft(x, y, hdg, gen):
    if gen < 50:
        num = 0
    elif gen >= 50 and gen < 100:
        num = (gen - 50) / 100
    else:
        num = 0.5
    
    return x < num and y < num

def rule_bottomright(x, y, hdg, gen):
    if gen < 50:
        num = 0
    elif gen >= 50 and gen < 100:
        num = (gen - 50) / 100
    else:
        num = 0.5
    
    return x > num and y < num

def main():
    # Initialize simulations specs.
    sim_topleft = {"number_of_creatures": 50, \
        "evolution_rule": rule_topleft, "brain_hidden_layers": (8, 8, 8), \
        "brain_activation_functions": ActivationFunction.Relu}
    sim_topright = {"number_of_creatures": 50, \
        "evolution_rule": rule_topright, "brain_hidden_layers": (8, 8, 8), \
        "brain_activation_functions": ActivationFunction.Linear}
    sim_bottomleft = {"number_of_creatures": 50, \
        "evolution_rule": rule_bottomleft, "brain_hidden_layers": (8, 8, 8), \
        "brain_activation_functions": ActivationFunction.BinaryStep}
    sim_bottomright = {"number_of_creatures": 50, \
        "evolution_rule": rule_bottomright, "brain_hidden_layers": (8, 8, 8), \
        "brain_activation_functions": ActivationFunction.Softplus}
    simulation_specs = (sim_topleft, sim_topright, sim_bottomleft, \
        sim_bottomright)
    
    # Start rendering window.
    window = Window(window_dimensions=(1920, 1350), \
        simulation_specs=simulation_specs, enable_multithreading=True)
    window.start(fps=60)
    window.quit()

if __name__ == "__main__":
    main()