import numpy as np

from neural_network import NeuralNetwork
from neural_network import ActivationFunction

from window import Window

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
    # Initialize simulations specs (just some random examples).
    sim_topleft = {"number_of_creatures": 50, \
        "evolution_rule": rule_topleft, "brain_hidden_layers": (8, 8, 8), \
        "brain_activation_functions": ActivationFunction.Relu, \
        "delta_time": 0.1, "max_time": 15.0, "mutation_max": 0.5}
    sim_topright = {"number_of_creatures": 50, \
        "evolution_rule": rule_topright, "brain_hidden_layers": (8, 8), \
        "brain_activation_functions": ActivationFunction.Linear, \
        "delta_time": 0.1, "max_time": 10.0}
    sim_bottomleft = {"number_of_creatures": 50, \
        "evolution_rule": rule_bottomleft, "brain_hidden_layers": (8,), \
        "brain_activation_functions": ActivationFunction.BinaryStep, \
        "delta_time": 0.1, "max_time": 7.5}
    sim_bottomright = {"number_of_creatures": 50, \
        "evolution_rule": rule_bottomright, "brain_hidden_layers": (8,), \
        "brain_activation_functions": ActivationFunction.Softplus, \
        "delta_time": 0.05, "max_time": 5.0, "max_creature_velocity": 0.5}
    simulation_specs = (sim_topleft, sim_topright, sim_bottomleft, \
        sim_bottomright)
    
    # Start rendering window.
    window = Window(window_dimensions=(1920, 1400), \
        simulation_specs=simulation_specs, enable_multithreading=False, \
        mutation_chance=0.01, mutation_max=0.05, brain_hidden_layers=(8,), \
        brain_activation_functions=ActivationFunction.Relu, delta_time=0.01, \
        max_time=10.0, max_creature_velocity=0.25, generations_per_render=1, \
        render_speedup=1.0, graph_initial_span=100, graph_max_span=int(1e9), \
        graph_max_number_of_ticks=8)
    window.start(fps=60)
    window.quit()

if __name__ == "__main__":
    main()