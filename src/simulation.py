import copy
import math
import multiprocessing as mp
import pickle
import random as rnd
import time

import numpy as np
import pygame

from neural_network import NeuralNetwork
from neural_network import ActivationFunction

class Environment:
    """
    This class holds the environment and the creatures within it. From this 
    class the environment is rendered and the creatures are updated and 
    rendered.
    """
    
    def __init__(self, number_of_creatures, optional_parameters={}):
        """
        Initialize creatures list and set several member variables. The 
        number of creatures is expected to be a positive integer, as is 
        assured when environments are created in the window class. The 
        optional parameters include (see window class __init__() method for 
        details):
            mutation_chance
            mutation_max
            brain_hidden_layers
            brain_activation_functions
        """
        # Set member variables.
        self.number_of_creatures = number_of_creatures
        
        # Set mutation member variables.
        self.mutation_chance = optional_parameters["mutation_chance"] \
            if "mutation_chance" in optional_parameters.keys() else 0.01
        self.mutation_max = optional_parameters["mutation_max"] \
            if "mutation_max" in optional_parameters.keys() else 0.05
        
        # Set brain size and activation functions member variables.
        self.brain_hidden_layers = \
            optional_parameters["brain_hidden_layers"] \
            if "brain_hidden_layers" in optional_parameters \
            else (8,)
        self.brain_activation_functions = \
            optional_parameters["brain_activation_functions"] \
            if "brain_activation_functions" in optional_parameters.keys() \
            else ActivationFunction.Relu
        
        # Initialize creatures list.
        self.creatures = []
        for _ in range(number_of_creatures):
            creature = Creature(brain_hidden_layers=self.brain_hidden_layers, \
                brain_activation_functions=self.brain_activation_functions)
            self.creatures.append(creature)
    
    def simulate_creature(self, creature, delta_time, max_time, \
        multi_threaded=False, shared_dict=None):
        """
        Takes as inputs the creature class, the delta time to use, the 
        maximum time to go until, and if this is a call from a multi threaded 
        setup or not. Simulates the creature by getting the position and 
        heading, feeding this data through its brain, and updating the 
        position and heading using the output of its brain until the current 
        time reaches the maximum time. Returns a lists containing the 
        position, velocity, and heading at each timestep in case of a single 
        threaded call, and sets a dictionary entry in case of a multi 
        threaded call.
        """
        # Initialize list of positions.
        positions = []
        
        # Run simulation.
        current_time = 0
        while current_time < max_time:
            # Get position.
            pos = creature.position
            
            # Get direction.
            hdg = creature.heading
            
            # Calculate north/south and east/west.
            n_s = math.sin(hdg)
            e_w = math.cos(hdg)
            
            # Feed forward brain.
            brain_output = creature.brain.feed_forward((pos[0], pos[1], \
                n_s, e_w))
            
            # Get velocity and heading delta.
            velocity = brain_output[0]
            heading_delta = brain_output[1]
            
            # Update creature velocity and heading.
            creature.velocity = clamp(velocity, 0.0, 0.3)
            creature.heading += heading_delta * delta_time
            
            # Make sure the heading is in the interval [0, 2pi).
            creature.heading %= 2 * math.pi
            
            # Update creature position.
            pos_delta_x = creature.velocity * math.cos(creature.heading) * delta_time
            pos_delta_y = creature.velocity * math.sin(creature.heading) * delta_time
            creature.position[0] += pos_delta_x
            creature.position[1] += pos_delta_y
            
            # Clamp position so the creatures do not exit the environment.
            creature.position[0] = clamp(creature.position[0], -1.0, 1.0)
            creature.position[1] = clamp(creature.position[1], -1.0, 1.0)
            
            # Add position to positions list.
            positions.append((current_time, \
                copy.deepcopy((creature.position, creature.velocity, creature.heading))))
            
            # Increment current time.
            current_time += delta_time
        
        # If this is a single threaded call, return the creature position data.
        if not multi_threaded:
            return positions
        # If this is a multi threaded call, do something esle.
        else:
            shared_dict[creature.id] = positions
    
    def simulate(self, delta_time, max_time=10, debug=False):
        """
        Populates a dictionary with the data returned from the simulate 
        function for each of the creatures. Returns the dictionary afterwards.
        This is a single threaded function.
        """
        if debug:
            print("Running simulation...")
        
        # Create dict for creature data.
        creature_data_dict = {}
        
        # Loop over all creatures and simulate each one.
        for creature in self.creatures:
            creature_data = self.simulate_creature(creature=creature, \
                delta_time=delta_time, max_time=max_time, \
                multi_threaded=False)
            creature_data_dict[creature.id] = creature_data
        
        if debug:
            print(f"Done.\n")
        
        # Return creature data.
        #return time_data, creature_data
        return creature_data_dict
    
    def simulate_multi_threaded(self, delta_time, max_time=10, \
        num_threads=4, debug=False):
        """
        Populates a dictionary with the data returned from the simulate 
        function for each of the creatures. Returns the dictionary afterwards.
        This is a multi threaded function.
        DEPRECATED: Multithreading creatures is not supported at this time. 
            It might be reimplemented in the future.
        """
        if debug:
            print(f"Running simulation (using {num_threads} threads)...")
        
        # Create shared dict.
        manager = mp.Manager()
        creature_data_dict = manager.dict()
        
        # Create list of arguments to starmap.
        starmap_args = []
        for creature in self.creatures:
            starmap_args.append((creature, delta_time, max_time, True, \
                creature_data_dict))
        
        # Starmap simulation of all creatures.
        with mp.Pool(processes=num_threads) as pool:
            pool.starmap(self.simulate_creature, starmap_args)
        
        if debug:
            print(f"Done.\n")
        
        # Return creature data.
        return creature_data_dict
    
    def apply_evolution(self, rule, generation, debug=False):
        """
        Checks which creatures have survived according to the rule function, 
        which accepts pos_x, pos_y, the heading, and the generation number. 
        This list then gets shuffled and new creatures are created and added 
        to an empty list until the starting number of creatures has been 
        reached. The creatures list is then replaced with this new list.
        """
        if debug:
            print("Applying evolution...")
        
        # Get list of all survived creatures.
        survivors = []
        for creature in self.creatures:
            pos_x = creature.position[0]
            pos_y = creature.position[1]
            hdg   = creature.heading
            if rule(pos_x, pos_y, hdg, generation):
                survivors.append(creature)
        
        # Shuffle the list.
        rnd.shuffle(survivors)
        number_of_survivors = len(survivors)
        
        # Initialize new creatures list.
        next_gen_creatures = []
        
        # If the number of survivors is zero, create a new batch of creatures.
        if number_of_survivors == 0:
            while len(next_gen_creatures) < self.number_of_creatures:
                # Create new creature.
                new_creature = Creature(\
                    brain_hidden_layers=self.brain_hidden_layers, \
                    brain_activation_functions=self.brain_activation_functions)
                
                # Add new creature to next gen creatures list.
                next_gen_creatures.append(new_creature)
        # Else apply the evolution algorithm on the surviving creatures.
        else:
            # Creature new creatures.
            next_parent_list_index = 0
            while len(next_gen_creatures) < self.number_of_creatures:
                # Create new creature.
                new_creature = Creature(parent=survivors[next_parent_list_index], \
                    mutation_chance=self.mutation_chance, \
                    mutation_max=self.mutation_max)
                
                # Add new creature to next gen creatures list.
                next_gen_creatures.append(new_creature)
                
                # Increment list index.
                next_parent_list_index += 1
                next_parent_list_index %= number_of_survivors
        
        # Set creature list to next gen creatures list.
        self.creatures = next_gen_creatures
        
        if debug:
            survive_percent = number_of_survivors / self.number_of_creatures * 100
            print(f"Number of survivors: {number_of_survivors}/" \
                f"{self.number_of_creatures} " \
                f"({survive_percent:.2f}%)")
            
            print("Done.\n")
        
        return number_of_survivors
    
    def render(self, display, window_dimensions, creatures, layout, index):
        """
        Renders the field and the creatures using the data supplied in the 
        creatures parameter. The layout parameter specifies the grid 
        dimensions and the index specifies the index of this simulation. The 
        first index is top left, the second to the right of that, etc, row by 
        row.
        """
        # Calculate grid margin and field size. The margin should always be 
        # 5% of the field size, centered vertically if necessary.
        # Equation used: width = margin + (field + margin) * layout[0]
        #   with margin = 0.05 * field
        available_width = 1100
        field_size = available_width / (1.05 * layout[0] + 0.05)
        margin = 0.05 * field_size
        
        # Calculate row and column number of this field.
        field_row = index // layout[0]
        field_column = index % layout[0]
        
        # Calculate x of this field.
        field_x = (window_dimensions[0] - available_width) / 2 \
            + (margin + field_size) * field_column + margin
        
        # Calculate y of this field (see readme for more information).
        layout_mid = (layout[1] - 1) / 2
        row_dist = layout_mid - field_row
        if layout[1] % 2 == 0:
            if row_dist > 0:
                field_y = window_dimensions[1] / 2 - margin / 2 - field_size \
                    - (row_dist - 0.5) * (margin + field_size)
            if row_dist < 0:
                field_y = window_dimensions[1] / 2 + margin / 2 \
                    - (row_dist + 0.5) * (margin + field_size)
        else:
            if row_dist == 0:
                field_y = window_dimensions[1] / 2 - field_size / 2
            if row_dist > 0:
                field_y = window_dimensions[1] / 2 - field_size / 2 \
                    - row_dist * (margin + field_size)
            if row_dist < 0:
                field_y = window_dimensions[1] / 2 + field_size / 2 \
                    + margin - (row_dist + 1) * (margin + field_size)
        
        # Render environment.
        field_color = (0, 100, 20)
        pygame.draw.rect(display, field_color, \
            (field_x, field_y, field_size, field_size))
        
        # Set creature size relative to field size.
        creature_size_inside  = field_size * 0.02
        creature_size_outside = creature_size_inside + 2 \
            if creature_size_inside >= 10 else creature_size_inside + 1
        
        # Render creatures.
        creature_color_inside = (120, 120, 120)
        creature_color_border = (40, 40, 40)
        for creature in creatures:
            # Get creature properties from tuple.
            creature_position, creature_velocity, creature_heading = creature
            
            # Get screen coordinates.
            screen_x = field_x + field_size / 2 + creature_position[0] \
                * (field_size / 2)
            screen_y = field_y + field_size / 2 - creature_position[1] \
                * (field_size / 2)
            
            # Render creature.
            pygame.draw.circle(display, creature_color_border, \
                (screen_x, screen_y), int(creature_size_outside))
            pygame.draw.circle(display, creature_color_inside, \
                (screen_x, screen_y), int(creature_size_inside))

class Creature:
    """
    This class holds all properties of the creature, such as:
        - Position (x, y)
        - Heading (angle)
        - Velocity (v)
        - Neural network brain
        - Generation number
        - Parent id
    """
    NEXT_ID = 0
    
    def __init__(self, brain_hidden_layers=None, \
        brain_activation_functions=None, parent=None, \
        mutation_chance=None, mutation_max=None):
        """
        Initialize id, random position, random heading, velocity of 0, 
        generation number depending on if a parent was specified or not, 
        parent id if a parent was specified, and a new brain or brain derived 
        from the parent also depending on if a parent was specified or not.
        
        Brain hidden layers and brain activation functions are required if no 
        parent was specified.
        
        Brain hidden layers specifies the number of nodes in each hidden 
        layer, the input layer is fixed at 4 nodes, and the output layer is 
        fixed at 2 nodes. For example, if brain hidden layers is equal to 
        (6, 5), network layout will be (4, 6, 5, 2).
        
        Brain activation functions can either be a single activation function 
        or a tuple of activation functions. If a single activation function 
        is specified, it will be used for all layers except the output layer, 
        which will have a linear activation function, and the input layer, 
        which has no activation function (aka a linear activation function). 
        If a tuple of activation functions is specified, it will be used 
        directly.
        
        Mutation chance and mutation max are required if a parent was 
        specified and denote the probability that any weight is altered at 
        all and the maximum percentage that any altered weight is adjusted 
        by, respectively.
        """
        # Check types.
        func_name = "Creature.__init__()"
        
        # Check if a correct set of parameters was specified.
        if not parent is None:
            assert not mutation_chance is None and not mutation_max is None, \
                f"{func_name}: If a parent is specified, mutation chance " \
                "and mutation max must also be specified."
        else:
            assert not brain_hidden_layers is None \
                and not brain_activation_functions is None, f"{func_name}: " \
                "If no parent is specified brain hidden layers and brain " \
                "activation functions must be specified."
        
        # Initialize id.
        self.id = Creature.NEXT_ID
        Creature.NEXT_ID += 1
        
        # Initialize random properties if no parent was specified.
        if parent is None:
            self.position = list(np.random.rand(2) * 2 - 1)
            self.heading  = np.random.rand(1)[0] * math.pi * 2
            self.velocity = 0
            self.generation_number = 1
            self.parent_id = -1
            brain_layers = tuple([4] + list(brain_hidden_layers) + [2])
            if isinstance(brain_activation_functions, ActivationFunction):
                brain_functions = tuple([brain_activation_functions] \
                    * len(brain_hidden_layers) + [ActivationFunction.Linear])
            else:
                brain_functions = brain_activation_functions
            self.brain = NeuralNetwork(brain_layers, brain_functions)
        # Initialize random properties, but a generation number, parent id, 
        # and brain depending on the parent specified, if a parent was 
        # specified.
        else:
            self.position = list(np.random.rand(2) * 2 - 1)
            self.heading  = np.random.rand(1)[0] * math.pi * 2
            self.velocity = 0
            self.generation_number = parent.generation_number + 1
            self.parent_id = parent.id
            self.brain = parent.brain.generate_variation(\
                mutation_chance=mutation_chance, mutation_max=mutation_max)

def clamp(value, minimum, maximum):
    """
    Clamp the value between the minimum and maximum.
    """
    if value < minimum:
        return minimum
    if value > maximum:
        return maximum
    return value
