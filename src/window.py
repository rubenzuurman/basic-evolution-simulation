import math
import multiprocessing as mp
import time

import pygame

from simulation import Environment
from simulation import ActivationFunction

def evo_rule(pos_x, pos_y, heading, generation):
    #return pos_x > -0.25 and pos_x < 0.25 and pos_y > -0.5 and pos_y < 0.5
    #return pos_x ** 2 + pos_y ** 2 < 0.5 ** 2
    if generation < 250:
        portion = 0.2
        return pos_x < -portion or pos_x > portion
    elif generation >= 250 and generation <= 500:
        portion = 0.2 + 0.7 * (generation - 250) / 250
        return pos_x < -portion or pos_x > portion
    else:
        portion = 0.9
        return pos_x < -portion or pos_x > portion

class SimulationData:
    """
    This class holds the environment (which holds the creatures), the data of 
    an entire simulation after running it, the data of one timestep while 
    rendering the simulation, a list containing the number of survivors per 
    generation, and a variable stating whether the simulation is done 
    rendering or not.
    """
    
    def __init__(self, environment, rule):
        """
        Environment is an instance of the simulation.Environment class. This 
        function initializes the time data list, creature data list, the list 
        containing the current creature data (this is the data that is 
        rendered and it only spans one timestep), and the list containing the 
        number of survivors per generation.
        """
        # Check types.
        func_name = "SimulationData.__init__()"
        
        # Check if environment is an instance of simulation.Environment.
        assert isinstance(environment, Environment), f"{func_name}: " \
            "Environment must be an instance of simulation.Environment."
        
        # Check if rule is callable.
        assert callable(rule), f"{func_name}: Rule must be callable."
        
        # Set environment.
        self.environment = environment
        
        # Set rule.
        self.rule = rule
        
        # Initialize creature data dict, this dict contains keys which are 
        # creature ids, and values which are tuples of the form 
        # (simulation_time, (position, velocity, heading)).
        self.creature_data_dict = {}
        
        # Set current creature data (this is the data that is rendered to the 
        # screen, spans one timestep).
        self.current_creature_data = []
        
        # Set survivors accumulate list (contains the number of survivors per 
        # generation).
        self.survivors_accumulate = []
        
        # Set boolean indicating if this simulation is currently being 
        # rendered.
        self.is_rendering = False
        
        self.temp = False
    
    def simulate_and_evolve(self, delta_time, max_time, generation):
        """
        Runs one full simulation round using the delta_time and max_time and 
        applies the evolution mechanism using the simulation rule function 
        and generation number.
        """
        # Check types.
        func_name = "SimulationData.__init__()"
        
        # Check if max time is greater or equal to than delta time.
        assert max_time >= delta_time, f"{func_name}: Max time must be " \
            "greater than or equal to delta time."
        
        # Run simulation.
        self.creature_data_dict = self.environment.simulate(\
            delta_time=delta_time, max_time=max_time)
        
        # Apply evolution.
        num_survivors = self.environment.apply_evolution(rule=self.rule, \
            generation=generation)
        self.survivors_accumulate.append(num_survivors)
    
    def simulate_and_evolve_multi_threaded(self, delta_time, max_time, \
        generation):
        """
        Runs one full simulation round using the delta_time and max_time and 
        applies the evolution mechanism using the simulation rule function 
        and generation number. This is the multi threaded version.
        """
        # Check types.
        func_name = "SimulationData.simulate_and_evolve_multi_threaded()"
        
        # Check if max time is greater or equal to than delta time.
        assert max_time >= delta_time, f"{func_name}: Max time must be " \
            "greater than or equal to delta time."
        
        # Run simulation.
        self.creature_data_dict = self.environment.simulate(\
            delta_time=delta_time, max_time=max_time)
        
        # Apply evolution.
        num_survivors = self.environment.apply_evolution(rule=self.rule, \
            generation=generation)
        self.survivors_accumulate.append(num_survivors)
        
        # Return resulting data.
        return [self.environment, self.creature_data_dict, \
            self.survivors_accumulate]
    
    def update_current_data(self, simulation_time, render_speedup=1):
        """
        Sets the current creature data list equal to creature data 
        corresponding to the simulation time specified. Sets the is_rendering 
        variable to false if the simulation is greater than the maximum time 
        in the time data list.
        """
        # Clear current creature data (intermediate variable to make sure the 
        # last state is rendered between renders).
        current_creature_data_in_loop = []
        
        # Loop over all creatures in the dict.
        simulation_finished = True
        for creature_id, creature_data in self.creature_data_dict.items():
            for step_time, timestep_data in creature_data:
                if step_time > simulation_time * render_speedup:
                    # Add tuple to current creature data list.
                    current_creature_data_in_loop.append(timestep_data)
                    simulation_finished = False
                    break
        
        # Update member variable if new creature data was fetched.
        if len(current_creature_data_in_loop) > 0:
            self.current_creature_data = current_creature_data_in_loop
        
        # Check if the simulation has finished.
        if simulation_finished:
            self.creature_data_dict = {}
            self.is_rendering = False

class Window:
    
    def __init__(self, window_dimensions, simulation_specs, \
        enable_multithreading=False):
        """
        Initializes pygame and creates a font and a display. It creates 
        simulations based on the provided simulation specs.
        Simulation specs is expected to be a tuple of dictionaries. Each of 
        these dictionaries contains the specifications for a simulation. The 
        following keys are implemented at this point:
            Required parameters
            number_of_creatures
                The number of creatures in this simulation.
            evolution_rule
                A function accepting four parameters (pos x, pos y, heading, 
                generation) and returning a boolean specifying if a creature 
                with this combination of coordinates, heading, and generation 
                will survive.
            
            Optional parameters
            mutation_chance (default=0.01)
                Float in the interval [0, 1] specifying the probility that 
                any individual weight will be altered in the evolution step.
            mutation_max (default=0.05)
                Float specifying the maximum amount a weight can be altered 
                by, proportional to the current value of the weight. E.g. if 
                this parameter is equal to 0.05 a weight will be multiplied 
                by a float in the interval [0.95, 1.05].
            brain_hidden_layers (default=(8,))
                A tuple containing the number of nodes in each hidden layer 
                of the brain of the creature, the input layer is fixed at 4 
                nodes, and the output layer is fixed at 2 nodes. E.g. if this 
                parameter is equal to (8, 6), the brain structure will look 
                like (4, 8, 6, 2).
            brain_activation_functions (default=ActivationFunction.Relu)
                (requires brain_hidden_layers to be defined)
                Controls the activation functions used by creature brains. 
                Must either be equal to a single ActivationFunction, or a 
                tuple of ActivationFunction's. If a single ActivationFunction 
                is supplied, it is used in every layer except for the output 
                layer, which will use a linear activation function. If a 
                tuple is supplied, the size of the tuple must be equal to the 
                number of layers minus 1 (because the input layer has no 
                activation function).
        """
        # Check types.
        func_name = "Window.__init__()"
        
        # Check if window_dimensions is a tuple.
        assert isinstance(window_dimensions, tuple), f"{func_name}: Window " \
            "dimensions must be a tuple containing two integers."
        
        # Check if window_dimensions has a length of 2.
        assert len(window_dimensions) == 2, f"{func_name}: Window " \
            "dimensions must be a tuple containing two integers."
        
        # Check if window_dimensions contains only integers.
        for n in window_dimensions:
            assert isinstance(n, int), f"{func_name}: Window dimensions " \
                "must be a tuple containing two integers."
        
        # Check if simulation specs is a tuple containing dicts.
        assert isinstance(simulation_specs, tuple), f"{func_name}: " \
            "Simulation specs must be a tuple of dictionaries."
        
        for index, spec_dict in enumerate(simulation_specs):
            # Make sure the entry is a dictionary.
            assert isinstance(spec_dict, dict), f"{func_name}: Simulation " \
                "specs must be a tuple of dictionaries."
            
            # Assert required entries.
            assert "number_of_creatures" in spec_dict.keys(), \
                f"[Simulation {index}] Specification dictionary must " \
                "contain 'number_of_creatures'."
            assert isinstance(spec_dict["number_of_creatures"], int), \
                f"[Simulation {index}] Number of creatures must be a " \
                "positive integer."
            assert spec_dict["number_of_creatures"] > 0, \
                f"[Simulation {index}] Number of creatures must be a " \
                "positive integer."
            
            assert "evolution_rule" in spec_dict.keys(), \
                f"[Simulation {index}] Specification dictionary must " \
                "contain 'evolution_rule'."
            assert callable(spec_dict["evolution_rule"]), \
                f"[Simulation {index}] Evolution rule must be callable."
            
            # Assert optional entries are of the correct type.
            if "mutation_chance" in spec_dict.keys():
                assert isinstance(spec_dict["mutation_chance"], float), \
                    f"[Simulation {index}] Mutation chance must be a float " \
                    "in the interval [0, 1]."
                assert spec_dict["mutation_chance"] >= 0 \
                    and spec_dict["mutation_chance"] <= 1, \
                    f"[Simulation {index}] Mutation chance must be a float " \
                    "in the interval [0, 1]."
            if "mutation_max" in spec_dict.keys():
                assert isinstance(spec_dict["mutation_max"], float), \
                    f"[Simulation {index}] Mutation max must be a float."
            if "brain_hidden_layers" in spec_dict.keys():
                assert isinstance(spec_dict["brain_hidden_layers"], tuple), \
                    f"[Simulation {index}] Brain hidden layers must be a " \
                    "tuple containing positive integers."
                for n in spec_dict["brain_hidden_layers"]:
                    assert isinstance(n, int), f"[Simulation {index}] " \
                        "Brain hidden layers must be a tuple containing " \
                        "positive integers."
                    assert n > 0, f"[Simulation {index}] Brain hidden " \
                        "layers must be a tuple containing positive integers."
            if "brain_activation_functions" in spec_dict.keys():
                assert "brain_hidden_layers" in spec_dict.keys(), \
                    f"[Simulation {index}] Brain hidden layers must be " \
                    "defined in order to use brain activation functions."
                baf = spec_dict["brain_activation_functions"]
                assert isinstance(baf, ActivationFunction) \
                    or isinstance(baf, tuple), f"[Simulation {index}] " \
                    "Brain activation functions must be an " \
                    "ActivationFunction or a tuple of ActivationFunction's."
                if isinstance(baf, tuple):
                    for a in baf:
                        assert isinstance(a, ActivationFunction), \
                            f"[Simulation {index}] Brain activation " \
                            "functions must be an ActivationFunction or a " \
                            "tuple of ActivationFunction's."
        
        # Check if the number of simulations is no bigger than 2500.
        assert len(simulation_specs) <= 10000, f"{func_name}: More than " \
            "10000 simulations are not supported."
        
        # Report unsupported simulation specification dictionary keys.
        supported_keys = ["number_of_creatures", "evolution_rule", \
            "mutation_chance", "mutation_max", "brain_hidden_layers", \
            "brain_activation_functions"]
        for index, spec_dict in enumerate(simulation_specs):
            for key in spec_dict.keys():
                if not key in supported_keys:
                    print(f"[WARNING] [Simulation {index}] Unsupported " \
                        f"key '{key}'.")
        
        # Initialize pygame.
        pygame.init()
        pygame.font.init()

        # Initialize font.
        self.font = pygame.font.SysFont("Courier New", 16)
        
        # Create display.
        self.window_dimensions = window_dimensions
        self.display = pygame.display.set_mode(window_dimensions, \
            flags=pygame.HWSURFACE | pygame.DOUBLEBUF)
        pygame.display.set_caption("Simulation Program")
        
        # Initialize simulations.
        self.simulations = []
        for spec_dict in simulation_specs:
            # Get required parameters from dictionary.
            number_of_creatures = spec_dict["number_of_creatures"]
            evolution_rule = spec_dict["evolution_rule"]
            
            # Get optional parameters from dictionary.
            optional_parameters = {}
            if "mutation_chance" in spec_dict.keys():
                optional_parameters["mutation_chance"] = \
                    spec_dict["mutation_chance"]
            if "mutation_max" in spec_dict.keys():
                optional_parameters["mutation_max"] = \
                    spec_dict["mutation_max"]
            if "brain_hidden_layers" in spec_dict.keys():
                optional_parameters["brain_hidden_layers"] = \
                    spec_dict["brain_hidden_layers"]
                if "brain_activation_functions" in spec_dict.keys():
                    optional_parameters["brain_activation_functions"] = \
                        spec_dict["brain_activation_functions"]
            
            # Create environment.
            environment = Environment(number_of_creatures, \
                optional_parameters=optional_parameters)
            
            # Create simulation.
            sim = SimulationData(environment, evolution_rule)
            
            # Add simulation.
            self.simulations.append(sim)
        
        # Calculate total number of creatures.
        self.number_of_creatures = sum([sim.environment.number_of_creatures \
            for sim in self.simulations])
        
        # Set member variables.
        self.sim_delta_time =  0.1
        self.sim_max_time   = 10.0
        
        self.generation = 0
        self.generations_per_render = 5
        
        self.render_speedup = 4
        
        self.mt = enable_multithreading
        
        # Create list of 100 double numbers.
        lst = []
        for n in range(101):
            lst.extend([n, n])
        
        # Create layout tuple.
        # Iterates over tuples like this: (0, 0), (1, 0), (1, 1), (2, 1), 
        #   (2, 2), (3, 2), (3, 3), etc until the product of the numbers 
        #   inside the tuple is greater than or equal to the number of 
        #   simulations.
        for a, b in zip(lst[1:], lst[:-1]):
            if len(self.simulations) <= a * b:
                self.render_layout = (a, b)
                break
        
        # Initialize multiprocessing pool if enabled.
        self.pool = mp.Pool() if self.mt else None
    
    def start(self, fps=60):
        """
        Renders and updates the simulations, this is the main loop of the 
        program.
        """
        # Create clock.
        clock = pygame.time.Clock()
        
        # Initialize variable keeping track of the starting time of the 
        # rendering of the simulation.
        simulation_start = 0
        
        # Initialize variable stating whether simulations are being rendered 
        # or not.
        rendering_simulation = False
        
        # Start update and render loop.
        running = True
        while running:
            # Handle events.
            for event in pygame.event.get():
                # Detect quit event.
                if event.type == pygame.QUIT:
                    running = False
                
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
            
            # Fill display with black.
            self.display.fill((0, 0, 0))
            
            # Run simulation if simulations are not being rendered.
            if not rendering_simulation:
                if self.mt:
                    rendering_simulation = \
                        self.run_simulation_step_multi_threaded()
                else:
                    rendering_simulation = \
                        self.run_simulation_step()
                if rendering_simulation:
                    # Set simulation start time.
                    simulation_start = time.time()
                    
                    # Set all sims to rendering.
                    for sim in self.simulations:
                        sim.is_rendering = True
            # Render simulation if necessary.
            else:
                # Set simulation time for simulations.
                simulation_time = time.time() - simulation_start
                
                # Update current data in simulations.
                for sim in self.simulations:
                    sim.update_current_data(simulation_time, \
                        render_speedup=self.render_speedup)
                
                # Update rendering variable.
                rendering_simulation = True in [sim.is_rendering \
                    for sim in self.simulations]
                
                # Reset simulation start if no longer rendering.
                if not rendering_simulation:
                    simulation_start = 0
            
            # Render environment and creatures.
            for index, sim in enumerate(self.simulations):
                sim.environment.render(self.display, \
                    window_dimensions=self.window_dimensions, \
                    creatures=sim.current_creature_data, \
                    layout=self.render_layout, index=index)
            
            # Render debug text.
            self.render_text(f"Rendering simulation: " \
                f"{rendering_simulation}", (10, 10))
            self.render_text(f"Current number of creatures: " \
                f"{self.number_of_creatures}", (10, 30))
            time_since_start_render = 0 if simulation_start == 0 \
                else time.time() - simulation_start
            self.render_text(f"Time since rendering start: " \
                f"{time_since_start_render:.1f}", (10, 50))
            self.render_text(f"Generation {self.generation}", (10, 70))
            mt_string = "Enabled" if self.mt else "Disabled"
            mt_procs = f" ({self.pool._processes} threads)" if self.mt else ""
            self.render_text(f"Multithreading: {mt_string}{mt_procs}", \
                (10, 90))
            
            # Render survivors graph.
            self.render_survivor_graph()
            
            # Update display.
            pygame.display.flip()
            
            # Tick clock.
            clock.tick(fps)
    
    def quit(self):
        # Quit pygame.
        pygame.quit()
        
        # Close process pool.
        if not self.pool is None:
            self.pool.close()
            self.pool.join()
    
    def run_simulation_step(self):
        """
        Simulates and evolves all simulations. Returns a boolean indicating 
        whether or not to render the last simulation.
        """
        # Loop over all simulations.
        for sim in self.simulations:
            # Run simulation and evolve.
            sim.simulate_and_evolve(delta_time=self.sim_delta_time, \
                max_time=self.sim_max_time, generation=self.generation)
        
        # Render simulation if necessary.
        render_sim = self.generation % self.generations_per_render == 0
        
        # Increment generation after checking whether to render or not to 
        # make sure the 0th generation is always rendered.
        self.generation += 1
        
        # Return boolean.
        return render_sim
    
    def run_simulation_step_multi_threaded(self):
        """
        Simulates and evolves all simulations using multithreading. Returns a 
        boolean indicating whether or not to render the last simulation.
        """
        # Run every simulation update asynchonously and append the result 
        # object to a list.
        results = []
        for index, sim in enumerate(self.simulations):
            result = self.pool.apply_async(\
                sim.simulate_and_evolve_multi_threaded, \
                args=(self.sim_delta_time, self.sim_max_time, self.generation))
            results.append(result)
        
        # Update simulation member variables using result objects.
        for index, result in enumerate(results):
            self.simulations[index].environment = result.get()[0]
            self.simulations[index].creature_data_dict = result.get()[1]
            self.simulations[index].survivors_accumulate = result.get()[2]
        
        # Render simulation if necessary.
        render_sim = self.generation % self.generations_per_render == 0
        
        # Increment generation after checking whether to render or not to 
        # make sure the 0th generation is always rendered.
        self.generation += 1
        
        # Return boolean.
        return render_sim
    
    def render_text(self, text, position, color=(255, 255, 255)):
        """
        Renders the text with the topleft at the position specified. Uses the 
        specified color if specified, else uses white.
        """
        text_surface = self.font.render(text, False, color)
        self.display.blit(text_surface, position)
    
    def render_survivor_graph(self):
        # Define color set.
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), \
            (255, 0, 255), (0, 255, 255), (255, 255, 255), (255, 128, 0), \
            (255, 0, 128), (0, 255, 128), (128, 255, 0), (0, 128, 255), \
            (128, 0, 255)]
        
        # Define graph width.
        graph_width  = 1000
        graph_height = 100
        
        # Define minimum and maximum x and y for graph.
        min_x = (self.window_dimensions[0] - graph_width) / 2
        min_y = 25
        
        # Render outer rect.
        pygame.draw.rect(self.display, (40, 40, 40), \
            (min_x - 1, min_y - 1, graph_width + 2, graph_height + 2))
        pygame.draw.rect(self.display, (0, 0, 0), \
            (min_x, min_y, graph_width, graph_height))
        
        # Loop over all simulations.
        for index, sim in enumerate(self.simulations):
            # Generate datapoints.
            screen_points = []
            for x, y in enumerate(sim.survivors_accumulate):
                screen_x = int(min_x + x)
                screen_y = int(min_y + graph_height - 100 \
                    * (y / sim.environment.number_of_creatures))
                screen_points.append((screen_x, screen_y))
            
            # Render datapoints.
            if len(screen_points) >= 2:
                if index >= len(colors):
                    color = (255, 255, 255)
                else:
                    color = colors[index]
                pygame.draw.lines(self.display, color, closed=False, \
                    points=screen_points)
