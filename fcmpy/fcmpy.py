# internal to the class is 
#  - the matrix used for calulation
#  - the dict mapping Cn to concept name
#  - a list of concepts
#  - a list of relations
# only add, no delete 
#
import numpy as np
import itertools
import json
import networkx as nx
import math


class Concept:
    def __init__(self, node, name, wordcloud, type_):
        self.id = node
        self.name = name,
        self.wordcloud = wordcloud,
        self.adjacent = {}
        self.type = type_  # "input, state, or output"

    def __str__(self):
        return str(self.id) + ' adjacent: ' + str([x.id for x in self.adjacent])

    def add_neighbor(self, neighbor, weight=0):
        self.adjacent[neighbor] = weight

    def get_connections(self):
        return self.adjacent.keys()  

    def get_name(self):
        return self.name      

    def get_id(self):
        return self.id

    def get_weight(self, neighbor):
        return self.adjacent[neighbor]

    def set_type(self, type_):
        if type_ != "input" or type_ != "state" or type_ != "output":
            return "ERR: type must be input, output, or state. Setting to 'state'."
        else:
            self.type = type_

    def get_type(self):
        return self.type

    def to_json(self):
        return json.dumps(self, default=lambda o: o.__dict__)


class Edge:
    def __init__(self, id_, from_node, to_node, weight):
        self.id = id_
        self.from_node = from_node
        self.to = to_node
        self.weight = weight


class FCM:
    def __init__(self):
        self.vert_dict = {}  # dictionary of concepts
        self.connection_dict = {}  # dictionary of connections
        self.correlation_matrix = [[]]  # used for calculations & output
        self.ordered_concepts = np.array([])
        self.cosmos_connection = ""  # if we want to get/set data from cosmos, set a connection
        self.squasher = "tanh"  # an enum indicating which squasher to use
        self.low_bound = 0  # low bound for input values
        self.high_bound = 1  # high bound for input values
        self.fuzz_range = 2  # number of steps for inputs
        self.graph = nx.DiGraph()  # internal representation of the graph

    def __iter__(self):
        return iter(self.vert_dict.values())

    def add_concept(self, node):
        new_concept = node
        self.vert_dict[node.id] = new_concept
        self.ordered_concepts = np.append(self.ordered_concepts, [node.id])
        self.graph.add_node(node.id)

        if len(self.vert_dict) == 1:
            self.correlation_matrix = np.append(self.correlation_matrix, [0])
        else:
            new_h_row = np.tile(0, len(self.vert_dict) - 1)
            self.correlation_matrix = np.column_stack((self.correlation_matrix, new_h_row))
            new_v_row = np.tile(0, len(self.vert_dict))
            self.correlation_matrix = np.vstack((self.correlation_matrix, new_v_row))

        return new_concept        

    #  update the graph and correlation matrix
    def add_connection(self, from_node, to_node, weight):
        from_index = np.where(self.ordered_concepts == from_node.id)
        to_index = np.where(self.ordered_concepts == to_node.id)
        self.correlation_matrix[from_index[0][0]][to_index[0][0]] = weight
        self.graph.add_weighted_edges_from([(from_node.id, to_node.id, weight)])

    #  return the NetworkX representation of the graph -> https://networkx.org/
    def get_graph(self):
        return self.graph

    # pass in a dictionary of concept ids/values, handle clamping
    def calculate(self, input_dict, max_iterations):
        # input_weights are an array of weights used to run the shifiz
        input_weights = np.tile(0.0,len(self.vert_dict))
        # default clamps are false
        input_clamps = np.zeros(len(self.vert_dict), dtype=bool)
        counter = 0
        # order the input weights to correlate to the matrix
        for i in self.ordered_concepts:
            input_weights[counter] = input_dict.get(i).get("in")
            if input_dict.get(i).get("clamp"):
                input_clamps[counter] = True
            counter = counter + 1

        result = []
        input_ec = input_weights
        result.append(input_ec)  # input weights are always the first value
        # if an input is clamped, keep it at it's value
        for i in range(max_iterations):
            input_ec = np.tanh(input_ec - input_ec.dot(self.correlation_matrix))
            ec_count = 0
            for c in input_ec:
                if input_clamps[ec_count]:
                    input_ec[ec_count] = input_weights[ec_count]
                ec_count += 1
            result.append(input_ec)
            # Check the euclidean distance for the last 3 iterations. If it's less than the stability threshold, stop
            stable_threshold = 0.1  # TODO this should be a configuration variable
            if len(result) > 3:
                stable_check_1 = result[len(result)-1]
                stable_check_2 = result[len(result)-2]
                stable_check_3 = result[len(result)-3]
                dist_1 = math.dist(stable_check_1, stable_check_2)
                dist_2 = math.dist(stable_check_1, stable_check_3)
                if dist_1 < stable_threshold and dist_2 < stable_threshold:
                    break

            # TODO below
            # - update DF to be the output they want...
            # - 

        return result

    def set_squasher(self, squasher):
        if squasher == "tanh":
            self.squasher = "tanh"
        elif squasher == "sigmoid":
            self.squasher = "sigmoid"
        else:
            self.squasher = "tanh"

    def get_matplotlib_labels(self): # return 2 arrays, one with ordered c numbers, one with corresponding names
        c_numbers = []
        c_names = []
        counter = 0
        for i in self.ordered_concepts:
            print(i)
            current_concept = self.vert_dict.get(i)
            c_numbers.append("c" + str(counter))
            c_names.append(current_concept.name[0])
            counter = counter + 1

        return {"concept_numbers": c_numbers, "concept_names": c_names}

    def output_gremlin(self):
        print("gremy")
        # return a gremlin query that can be used to save to gremlin

    # return json that can be used for d3 graph
    def output_d3(self):
        print("d3")
        d3_json = {}
        nodes = []
        links = []
        correlations = np.array(self.correlation_matrix)
        row_counter = 0
        for r in correlations:
            from_concept = self.vert_dict[self.ordered_concepts[row_counter]]
            nodes.append({"name": from_concept.get_name()[0], "id": from_concept.get_id()})
            col_counter = 0
            for c in r:
                if c > 0:
                    to_concept = self.vert_dict[self.ordered_concepts[col_counter]]
                    links.append({"from": from_concept.get_id(), "to": to_concept.get_id(), "weight": c})
                col_counter += 1
            row_counter += 1
        d3_json["nodes"] = nodes
        d3_json["links"] = links
        return json.dumps(d3_json)

    # run every possible scenario for this model
    # (1) find all the possible input values given linguistic inputs
    # (2) create the cartesian product to get every possible input
    # (3) use these as inputs, run the model a ton    
    def run_all_possible_scenarios(self, max_iterations):
        low_val = self.low_bound
        high_val = self.high_bound
        fuzz_count = self.fuzz_range
        # initialize list will potential values in range
        list_1 = np.linspace(low_val, high_val, num=fuzz_count)
        # get the cartesian product to calculate all possible inputs
        input_concepts = {}
        for i in self.ordered_concepts:
            if self.vert_dict.get(i).get_type() == "in":
                input_concepts[i] = self.vert_dict.get(i)

        print("Input concept count:")
        print(str(len(input_concepts)))
        print("Total concept count:")
        print(str(len(self.vert_dict)))

        if len(input_concepts) < 1:
            raise Exception("You must have at least one input concept!")

        # unique_combinations = list(itertools.product(list_1, repeat=len(self.vert_dict)))
        # TODO return an error if there are no concepts marked as input
        unique_combinations = list(itertools.product(list_1, repeat=int(len(input_concepts))))

        print("Unique Combos")
        print(len(unique_combinations))

        # unique_combinations_slice = unique_combinations[:100]
        # unique_combinations = unique_combinations_slice

        # this is a hack to get through for now...
        only_five = []
        for u in unique_combinations:
            five_check = sum(u)
            if 0 < five_check < 6:
                only_five.append(u)

        unique_combinations = only_five
        print("Only fives:")
        print(len(unique_combinations))

        all_inputs = []
        for u in unique_combinations:
            counter = 0
            this_input = {}
            for i in self.ordered_concepts:
                # if this is an input clamp it
                if self.vert_dict.get(i).get_type() == "in":
                    this_input[i] = { "in": u[counter], "clamp": True }
                    counter = counter + 1
                else:
                    this_input[i] = { "in": 0, "clamp": False }
            all_inputs.append(this_input)
        all_possible_outputs = []
        for a in all_inputs:
            all_possible_outputs.append(self.calculate(a, max_iterations))
        
        return all_possible_outputs

    # Import CSV for matrix and for concepts
    def create_from_csv(self, correlation_matrix, node_definitions):
        # TODO add connections
        for node in node_definitions:
            c = Concept(node[1], node[0], "", node[2])
            new_concept = c
            self.vert_dict[c.id] = new_concept
            self.ordered_concepts = np.append(self.ordered_concepts, [c.id])
            self.graph.add_node(c.id)

        self.correlation_matrix = correlation_matrix

    def scenario_as_text(self):
        print("hm")
        # convert 
