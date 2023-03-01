import math
import multiprocessing as mp
import time

import pygame

from simulation import Environment
from simulation import ActivationFunction

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
    
    def simulate_and_evolve(self, generation):
        """
        Runs one full simulation and applies the evolution mechanism using 
        the simulation rule function and generation number.
        """
        # Run simulation.
        self.creature_data_dict = self.environment.simulate()
        
        # Apply evolution.
        num_survivors = self.environment.apply_evolution(rule=self.rule, \
            generation=generation)
        self.survivors_accumulate.append(num_survivors)
    
    def simulate_and_evolve_multi_threaded(self, generation):
        """
        Runs one full simulation and applies the evolution mechanism using 
        the simulation rule function and generation number. This is the multi 
        threaded version.
        """
        # Run simulation.
        self.creature_data_dict = self.environment.simulate()
        
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

def default_graph_color_generator(index):
    """
    Default graph color generator. Generates (in some sense) simple colors 
    until it runs out. After these are used up it generates somewhat random 
    colors.
    """
    # Define simple colors.
    simple_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), \
        (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0), (0, 0, 128), \
        (128, 128, 0), (128, 0, 128), (0, 128, 128)]
    
    # Return a simple color if possible.
    if index < len(simple_colors):
        return simple_colors[index]
    
    # Else return a somewhat random color.
    adjusted_index = index - len(simple_colors)
    return ((200 + adjusted_index * 25) % 255, \
        (100 + adjusted_index * 50) % 255, (adjusted_index * 100) % 255)

class Window:
    
    def __init__(self, window_dimensions, simulation_specs, \
        enable_multithreading=False, mutation_chance=0.01, \
        mutation_max=0.05, brain_hidden_layers=(8,), \
        brain_activation_functions=ActivationFunction.Relu, delta_time=0.1, \
        max_time=10.0, max_creature_velocity=0.25, generations_per_render=1, \
        render_speedup=1.0, graph_initial_span=100, graph_max_span=int(1e9), \
        graph_max_number_of_ticks=8, \
        graph_color_generator=default_graph_color_generator):
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
                Float in the interval [0, 1] specifying the probability that 
                any individual weight will be altered in the evolution step. 
                If not present, the simulation will use the global mutation 
                chance passed to the Window constructor.
            mutation_max (default=0.05)
                Float specifying the maximum amount a weight can be altered 
                by, proportional to the current value of the weight. E.g. if 
                this parameter is equal to 0.05 a weight will be multiplied 
                by a float in the interval [0.95, 1.05]. If not present, the 
                simulation will use the global mutation max passed to the 
                Window constructor.
            brain_hidden_layers (default=(8,))
                A tuple containing the number of nodes in each hidden layer 
                of the brain of the creature, the input layer is fixed at 4 
                nodes, and the output layer is fixed at 2 nodes. E.g. if this 
                parameter is equal to (8, 6), the brain structure will look 
                like (4, 8, 6, 2). If not present, the simulation will use 
                the global brain hidden layers passed to the Window 
                constructor.
            brain_activation_functions (default=ActivationFunction.Relu)
                Controls the activation functions used by creature brains. 
                Must either be equal to a single ActivationFunction, or a 
                tuple of ActivationFunction's. If a single ActivationFunction 
                is supplied, it is used in every layer except for the output 
                layer, which will use a linear activation function. If a 
                tuple is supplied, the size of the tuple must be equal to the 
                number of layers minus 1 (because the input layer has no 
                activation function). If not present, the simulation will use 
                the global brain activation functions passed to the Window 
                constructor.
            delta_time (default=0.1)
                Sets the time amount the simulation time is increased by 
                every timestep. If not present, the simulation will use the 
                global delta time passed to the Window constructor.
            max_time (default=10.0)
                When this simulation time is reached, the simulation 
                terminates and the evolution step starts. If not present, the 
                simulation will use the global max time passed to the Window 
                constructor.
            max_creature_velocity (default=0.25)
                The maximum creature velocity in half fields per second. Must 
                be a positive float. If not present, the simulation will use 
                the global max creature velocity passed to the Window 
                constructor.
        Enable multithreading must be a boolean specifying whether to use 
            multithreading or not.
        Mutation chance must be a float in the interval [0, 1], it works the 
            same as mutation_chance per simulation, but globally. Simulation 
            settings override global settings locally.
            Mutation max must be a positive float, it works the same as 
            mutation_max per simulation, but globally. Simulation settings 
            override global settings locally.
        Brain hidden layers must be a tuple containing the number of nodes in 
            each hidden layer of the brain of the creature. It works the same 
            as brain_hidden_layers per simulation, but globally. Simulation 
            settings override global settings locally.
        Brain activation functions controls the activation functions used by 
            creature brains. It works the same as brain_activation_functions 
            per simulation, but globally. Simulation settings override global 
            settings locally.
        Delta time must be a positive float, it works the same as delta_time 
            per simulation, but globally. Simulation settings override global 
            settings locally.
        Max time must be a positive float, it works the same as max_time per 
            simulation, but globally. Simulation settings override global 
            settings locally.
        Max creature velocity must be a positive float, it works the same as 
            max_creature_velocity per simulation, but globally. Simulation 
            settings override global settings locally.
        Generations per render must be a positive integer. It specifies the 
            number of generations to simulate before rendering. For example, 
            if this setting is set to 5, it will render every fifth 
            simulation.
        Render speedup must be a positive float. It specifies how fast the 
            simulations should be rendered. For example, a setting of 1.0 
            renders the simulation in real time, 0.5 at half the speed, and 
            2.0 at double the speed.
        Graph initial span specifies the number of generations that fit 
            inside the graph at the start of the program. Must be an integer 
            greater than or equal to 2.
        Graph max span specifies the number of generations that may fit 
            inside the graph before it switches to scrolling. Must be an 
            integer greater than or equal to 2.
        Graph max number of ticks specifies the maximum number of ticks to 
            display on the x axis of the graph. Must be a non-negative 
            integer. Set to 0 to show no ticks.
        Graph color generator is a function which takes the simulation index 
            as a parameter and return the color for that simulation. The 
            color white is used if no valid color is returned.
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
                    f"[Simulation {index}] Mutation max must be a " \
                    "non-negative float."
                assert spec_dict["mutation_max"] >= 0, \
                    f"[Simulation {index}] Mutation max must be a " \
                    "non-negative float."
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
                    
                    num_layers = len(spec_dict["brain_hidden_layers"]) + 2 \
                        if "brain_hidden_layers" in spec_dict \
                        else len(brain_hidden_layers) + 2
                    assert len(baf) == num_layers - 1, \
                        f"[Simulation {index}] Brain activation functions " \
                        "(as a tuple) must have a length equal to the " \
                        "total number of layers minus 1."
            
            if "delta_time" in spec_dict.keys():
                assert isinstance(spec_dict["delta_time"], float), \
                    f"[Simulation {index}] Delta time must be a positive " \
                    "float."
                assert spec_dict["delta_time"] > 0, f"[Simulation {index}] " \
                    "Delta time must be a positive float."
            if "max_time" in spec_dict.keys():
                assert isinstance(spec_dict["max_time"], float), \
                    f"[Simulation {index}] Max time must be a positive " \
                    "float."
                assert spec_dict["max_time"] > 0, f"[Simulation {index}] " \
                    "Max time must be a positive float."
            delta_time_temp = spec_dict["delta_time"] \
                if "delta_time" in spec_dict.keys() else delta_time
            max_time_temp   = spec_dict["max_time"] \
                if "max_time" in spec_dict.keys() else max_time
            if delta_time_temp >= max_time_temp:
                print(f"[WARNING] [Simulation {index}] Delta time is " \
                    "greater than or equal to max time, this may lead to " \
                    "the simulation only executing one timestep.")
            
            if "max_creature_velocity" in spec_dict.keys():
                assert isinstance(spec_dict["max_creature_velocity"], float), \
                    f"[Simulation {index}] Max creature velocity must be a " \
                    "positive float."
                assert spec_dict["max_creature_velocity"] > 0, \
                    f"[Simulation {index}] Max creature velocity must be a " \
                    "positive float."
        
        # Check if the number of simulations is no bigger than 10000.
        assert len(simulation_specs) <= 10000, f"{func_name}: More than " \
            "10000 simulations are not supported."
        
        # Check if enable_multithreading is a boolean.
        assert isinstance(enable_multithreading, bool), f"{func_name}: " \
            "Enable multithreading must be a boolean."
        
        # Check if mutation_chance is a float in the interval [0, 1].
        assert isinstance(mutation_chance, float), f"{func_name}: Mutation " \
            "chance must be a float in the interval [0, 1]."
        assert mutation_chance >= 0 and mutation_chance <= 1, \
            f"{func_name}: Mutation chance must be a float in the interval " \
            "[0, 1]."
        
        # Check if mutation_max is a non-negative float.
        assert isinstance(mutation_max, float), f"{func_name}: Mutation " \
            "max must be a non-negative float."
        assert mutation_max >= 0, f"{func_name}: Mutation " \
            "max must be a non-negative float."
        
        # Check if brain hidden layers is a tuple containing positive 
        # integers.
        assert isinstance(brain_hidden_layers, tuple), \
            f"{func_name}: Brain hidden layers must be a " \
            "tuple containing positive integers."
        for n in brain_hidden_layers:
            assert isinstance(n, int), f"{func_name}: Brain hidden layers " \
                "must be a tuple containing positive integers."
            assert n > 0, f"{func_name}: Brain hidden layers " \
                "must be a tuple containing positive integers."
        
        # Check if brain activation functions is either an ActivationFunction 
        # or a tuple containing ActivationFunction's.
        assert isinstance(brain_activation_functions, ActivationFunction) \
            or isinstance(brain_activation_functions, tuple), \
            f"{func_name}: Brain activation functions must be an " \
            "ActivationFunction or a tuple of ActivationFunction's."
        if isinstance(brain_activation_functions, tuple):
            for a in brain_activation_functions:
                assert isinstance(a, ActivationFunction), \
                    f"{func_name}: Brain activation functions must be an " \
                    "ActivationFunction or a tuple of ActivationFunction's."
            
            num_layers = len(brain_hidden_layers) + 2
            assert len(brain_activation_functions) == num_layers - 1, \
                f"{func_name}: Brain activation functions (as a tuple) " \
                "must have a length equal to the total number of layers " \
                "minus 1."
        
        # Check if delta_time and max_time are positive floats.
        assert isinstance(delta_time, float), f"{func_name}: Delta time " \
            "must be a positive float."
        assert delta_time > 0, f"{func_name}: Delta time " \
            "must be a positive float."
        assert isinstance(max_time, float), f"{func_name}: Max time " \
            "must be a positive float."
        assert max_time > 0, f"{func_name}: Max time " \
            "must be a positive float."
        
        # Check if max_creature_velocity is a positive float.
        assert isinstance(max_creature_velocity, float), f"{func_name}: " \
            "Max creature velocity must be a positive float."
        assert max_creature_velocity > 0, f"{func_name}: Max creature " \
            "velocity must be a positive float."
        
        # Check if generations_per_render is a positive integer.
        assert isinstance(generations_per_render, int), f"{func_name}: " \
            "Generations per render must be a positive integer."
        assert generations_per_render > 0, f"{func_name}: " \
            "Generations per render must be a positive integer."
        
        # Check if render_speedup is a positive float.
        assert isinstance(render_speedup, float), f"{func_name}: Render " \
            "speedup must be a positive float."
        assert render_speedup > 0, f"{func_name}: Render " \
            "speedup must be a positive float."
        
        # Check if graph_initial_span is a positive integer.
        assert isinstance(graph_initial_span, int), f"{func_name}: Graph " \
            "initial span must be an integer greater than or equal to 2."
        assert graph_initial_span >= 2, f"{func_name}: Graph " \
            "initial span must be an integer greater than or equal to 2."
        
        # Check if graph_max_span is an integer greater than or equal to 2.
        assert isinstance(graph_max_span, int), f"{func_name}: Graph " \
            "max span must be an integer greater than or equal to 2."
        assert graph_max_span >= 2, f"{func_name}: Graph " \
            "max span must be an integer greater than or equal to 2."
        
        # Check if graph_max_number_of_ticks is a non-negative integer.
        assert isinstance(graph_max_number_of_ticks, int), f"{func_name}: " \
            "Graph max number of ticks must be a non-negative integer."
        assert graph_max_number_of_ticks >= 0, f"{func_name}: " \
            "Graph max number of ticks must be a non-negative integer."
        
        # Check if graph color generator is callable.
        assert callable(graph_color_generator), f"{func_name}: " \
            "Graph color generator must be callable."
        
        # Warn the user if delta_time is greater than or equal to max_time.
        if delta_time >= max_time:
            print(f"[WARNING] Delta time is greater than or equal to max " \
                "time, this may lead to the simulation only executing one " \
                "timestep.")
        
        # Report unsupported simulation specification dictionary keys.
        supported_keys = ["number_of_creatures", "evolution_rule", \
            "mutation_chance", "mutation_max", "brain_hidden_layers", \
            "brain_activation_functions", "delta_time", "max_time", \
            "max_creature_velocity"]
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
        self.graph_label_font = pygame.font.SysFont("Courier New", 12)
        
        # Create display.
        self.window_dimensions = window_dimensions
        self.display = pygame.display.set_mode(window_dimensions, \
            flags=pygame.HWSURFACE | pygame.DOUBLEBUF)
        pygame.display.set_caption("Basic Evolution Simulation")
        
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
            else:
                optional_parameters["mutation_chance"] = mutation_chance
            if "mutation_max" in spec_dict.keys():
                optional_parameters["mutation_max"] = \
                    spec_dict["mutation_max"]
            else:
                optional_parameters["mutation_max"] = mutation_max
            
            if "brain_hidden_layers" in spec_dict.keys():
                optional_parameters["brain_hidden_layers"] = \
                    spec_dict["brain_hidden_layers"]
            else:
                optional_parameters["brain_hidden_layers"] = brain_hidden_layers
            if "brain_activation_functions" in spec_dict.keys():
                optional_parameters["brain_activation_functions"] = \
                    spec_dict["brain_activation_functions"]
            else:
                optional_parameters["brain_activation_functions"] = brain_activation_functions
            
            if "delta_time" in spec_dict.keys():
                optional_parameters["delta_time"] = spec_dict["delta_time"]
            else:
                optional_parameters["delta_time"] = delta_time
            if "max_time" in spec_dict.keys():
                optional_parameters["max_time"] = spec_dict["max_time"]
            else:
                optional_parameters["max_time"] = max_time
            
            if "max_creature_velocity" in spec_dict.keys():
                optional_parameters["max_creature_velocity"] = spec_dict["max_creature_velocity"]
            else:
                optional_parameters["max_creature_velocity"] = max_creature_velocity
            
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
        
        # Initialize generation counter member variable.
        self.generation = 0
        
        # Initialize generations per render and render speedup member 
        # variables.
        self.generations_per_render = generations_per_render
        self.render_speedup = render_speedup
        
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
        self.mt = enable_multithreading
        self.pool = mp.Pool() if self.mt else None
        
        # Set graph parameter member variables.
        self.graph_initial_span = graph_initial_span
        self.graph_max_span = graph_max_span
        self.graph_max_number_of_ticks = graph_max_number_of_ticks
        self.graph_color_generator = graph_color_generator
    
    def start(self, fps=60):
        """
        Renders and updates the simulations, this is the main loop of the 
        program.
        """
        # Check types.
        func_name = "Window.start()"
        
        # Check that fps is a positive integer.
        assert isinstance(fps, int), f"{func_name}: Fps must be a positive " \
            "integer."
        assert fps > 0, f"{func_name}: Fps must be a positive integer."
        
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
            sim.simulate_and_evolve(generation=self.generation)
        
        # Increment generation.
        self.generation += 1
        
        # Render simulation if necessary or if this was the first simulation.
        render_sim = self.generation % self.generations_per_render == 0 \
            or self.generation == 1
        
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
                args=(self.generation,))
            results.append(result)
        
        # Update simulation member variables using result objects.
        for index, result in enumerate(results):
            self.simulations[index].environment = result.get()[0]
            self.simulations[index].creature_data_dict = result.get()[1]
            self.simulations[index].survivors_accumulate = result.get()[2]
        
        # Increment generation.
        self.generation += 1
        
        # Render simulation if necessary or if this was the first simulation.
        render_sim = self.generation % self.generations_per_render == 0 \
            or self.generation == 1
        
        # Return boolean.
        return render_sim
    
    def render_text(self, text, position, color=(255, 255, 255)):
        """
        Renders the text with the topleft at the position specified. Uses the 
        specified color if specified, else uses white.
        """
        text_surface = self.font.render(text, False, color)
        self.display.blit(text_surface, position)
    
    def render_graph_text(self, text, position, center="both", \
        color=(255, 255, 255)):
        """
        Renders the text at the position specified. Uses the specified color 
        if specified, otherwise uses white. Uses the graph label font. 
        Centers in the y direction if center is 'vertical', in the x 
        direction if center is 'horizontal', and in both directions if center 
        is 'both'.
        """
        text_surface = self.graph_label_font.render(text, False, color)
        if center == "vertical":
            self.display.blit(text_surface, \
                (position[0], position[1] - text_surface.get_height() / 2))
        elif center == "horizontal":
            self.display.blit(text_surface, \
                (position[0] - text_surface.get_width() / 2, position[1]))
        elif center == "both":
            self.display.blit(text_surface, \
                (position[0] - text_surface.get_width() / 2, \
                    position[1] - text_surface.get_height() / 2))
        else:
            self.display.blit(text_surface, position)
            print("[WARNING] Window.render_graph_text(): Invalid value " \
                f"for center parameter '{center}'.")
    
    def render_survivor_graph(self):
        # Get number of generations from first simulation.
        num_gens = len(self.simulations[0].survivors_accumulate)
        
        # Reduce initial span if max span is smaller.
        initial_span = self.graph_initial_span
        max_span = self.graph_max_span
        if max_span < initial_span:
            initial_span = max_span
        
        # Calculate start and end index of the graph.
        if num_gens <= initial_span:
            start_index = 1
            end_index = initial_span
        else:
            if num_gens < max_span:
                start_index = 1
                end_index = num_gens
            else:
                start_index = num_gens - max_span + 1
                end_index = num_gens
        
        # Define graph width.
        graph_width  = 1000
        graph_height = 100
        
        # Define minimum and maximum x and y for graph.
        min_x = (self.window_dimensions[0] - graph_width) / 2
        min_y = 25
        
        # Render outer rect (the extra one is to create an interior of 
        # graph_width pixels wide).
        pygame.draw.rect(self.display, (40, 40, 40), \
            (min_x - 1, min_y - 1, graph_width + 2 + 1, graph_height + 2))
        pygame.draw.rect(self.display, (0, 0, 0), \
            (min_x, min_y, graph_width + 1, graph_height))
        
        # Render y axis ticks and labels.
        pygame.draw.line(self.display, (120, 120, 120), (min_x - 1, min_y - 1), \
            (min_x - 5, min_y - 1))
        self.render_graph_text("100%", (min_x - 35, min_y), \
            center="vertical", color=(120, 120, 120))
        
        pygame.draw.line(self.display, (120, 120, 120), (min_x - 1, min_y + graph_height / 2), \
            (min_x - 5, min_y + graph_height / 2))
        self.render_graph_text(" 50%", (min_x - 35, min_y + graph_height / 2), \
            center="vertical", color=(120, 120, 120))
        
        pygame.draw.line(self.display, (120, 120, 120), (min_x - 1, min_y + graph_height), \
            (min_x - 5, min_y + graph_height))
        self.render_graph_text("  0%", (min_x - 35, min_y + graph_height), \
            center="vertical", color=(120, 120, 120))
        
        # Render horizontal lines in the graph at 25%, 50%, and 75%.
        horizontal_line_y_coords = [graph_height / 4 * 1, \
            graph_height / 4 * 2, graph_height / 4 * 3]
        for line_y in horizontal_line_y_coords:
            pygame.draw.line(self.display, (40, 40, 40), \
                (min_x, min_y + line_y), \
                (min_x + graph_width, min_y + line_y))
        
        # Choose x axis ticks and labels.
        xticks = []
        xticks_delta_generator = lambda index: [1 * 10 ** (index // 3), \
            2 * 10 ** (index // 3), 5 * 10 ** (index // 3)][index % 3]
        for i in range(81):
            # If max ticks is equal to zero, break so the xticks list stays 
            # empty.
            if self.graph_max_number_of_ticks == 0:
                break
            
            # Get next tick size.
            tick_size = xticks_delta_generator(i)
            
            # Skip tick size if the numbers of ticks is at least bigger than 
            # the maximum number of ticks.
            if (end_index - start_index) // tick_size > self.graph_max_number_of_ticks:
                continue
            
            # Calculate start and end of the checkable region.
            check_start = (start_index // tick_size) * tick_size
            check_end   = (end_index // tick_size + 1) * tick_size
            
            # Get ticks inside visible region.
            xticks = [tick for tick \
                in range(check_start, check_end + tick_size, tick_size) \
                if tick >= start_index and tick <= end_index]
            
            # Add 1 to xticks if the start index is equal to 1.
            if start_index == 1:
                xticks = [1] + xticks
            
            # Calculate number of ticks and break if it is lower or equal to 
            # the maximum number of ticks.
            num_ticks = len(xticks)
            if num_ticks <= self.graph_max_number_of_ticks:
                break
        
        # Create function for calculating the x coordinate of a datapoint.
        # Coefficients calculated by drawing a line from (1, 1) to 
        # (end - start, graph_width + 1). This will create a mapping from 1 
        # to end - start to 1 to 1001, which can be shifted left by 1.
        a = graph_width / (end_index - start_index)
        b = 1 - a
        calculate_graph_x = lambda x: a * (x + 1 - start_index) + b - 1
        
        # Iterate over all xticks.
        for xtick in xticks:
            # Calculate screen x value of the tick.
            line_x = min_x + calculate_graph_x(xtick)
            
            # Render x axis ticks and labels.
            pygame.draw.line(self.display, (120, 120, 120), \
                (line_x, min_y + graph_height), \
                (line_x, min_y + graph_height + 5))
            self.render_graph_text(f"{xtick}", (line_x, min_y + graph_height + 8), \
                center="horizontal", color=(120, 120, 120))
            
            # Render vertical line.
            pygame.draw.line(self.display, (40, 40, 40), (line_x, min_y), \
                (line_x, min_y + graph_height))
        
        # Loop over all simulations.
        for index, sim in enumerate(self.simulations):
            # Generate datapoints.
            screen_points = []
            for x, y in enumerate(sim.survivors_accumulate):
                # Shift range from 0-gens-1 to 1-gens.
                x += 1
                
                # Check if this point is inside the graph.
                if x < start_index or x > end_index:
                    continue
                
                # Calculate screen coordinates of the point and add it to the 
                # list.
                screen_x = min_x + calculate_graph_x(x)
                screen_y = int(min_y + graph_height - 100 \
                    * (y / sim.environment.number_of_creatures))
                screen_points.append((screen_x, screen_y))
            
            # Render datapoints.
            if len(screen_points) >= 2:
                # Get color from color generator.
                color = self.graph_color_generator(index)
                
                # Check if color is valid.
                if not isinstance(color, tuple):
                    print(f"[WARNING] [Simulation {index}] Invalid color " \
                        f"was returned from color generator '{color}'.")
                    color = (255, 255, 255)
                if not len(color) == 3:
                    print(f"[WARNING] [Simulation {index}] Invalid color " \
                        f"was returned from color generator '{color}'.")
                    color = (255, 255, 255)
                for c in color:
                    if not isinstance(c, int):
                        print(f"[WARNING] [Simulation {index}] Invalid color " \
                            f"was returned from color generator '{color}'.")
                        color = (255, 255, 255)
                        break
                    if not (c >= 0 and c <= 255):
                        print(f"[WARNING] [Simulation {index}] Invalid color " \
                            f"was returned from color generator '{color}'.")
                        color = (255, 255, 255)
                        break
                
                # Render lines.
                pygame.draw.lines(self.display, color, closed=False, \
                    points=screen_points)
