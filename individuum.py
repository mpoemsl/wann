""" Class for an individuum that represents a valid neural network architecture via node layers, connection tables and activations. """

from utilities import ACTIVATION_DICT

import numpy as np
import pickle


class Individuum():


    def __init__(self, ratio_enabled=0.4, n_inputs=784, n_outputs=10, **kwargs):
        """ Initializes node layers, connection tables and activations."""

        input_layer = {
            "id": 0,
            "table_ix": 0,
            "size": n_inputs,
            "src": None,
            "dst": 1,
            "pos": 0
        }

        output_layer = {
            "id": 1,
            "table_ix": None,
            "size": n_outputs,
            "src": 0,
            "dst": None,
            "pos": 1
        }

        main_connection_table = (np.random.random((input_layer["size"], output_layer["size"])) < ratio_enabled).astype(bool)

        self.layers = [input_layer, output_layer]
        self.connection_tables = [main_connection_table]
        self.activations = np.ones(output_layer["size"])


    def save_to(self, path):
        
        with open(path + "_layers", "wb+") as fh:
            pickle.dump(self.layers, fh)

        for ix, table in enumerate(self.connection_tables):
            np.save(path + "_table_{}.npy".format(ix), table)

        np.save(path + "_activations.npy", self.activations)


    def load_from(self, path):
         
        with open(path + "_layers", "rb") as fh:
            self.layers = pickle.load(fh)

        self.connection_tables = []

        for ix in range(len(self.layers) - 1):
            self.connection_tables.append(np.load(path + "_table_{}.npy".format(ix)))

        self.activations = np.load(path + "_activations.npy")


    def add_node(self):
        """ Splits a random enabled connections and places a node inbetween by adding a node layer if necessary. """

        split_src_layer_id, split_src_node, split_dst_layer_id, split_dst_node = self.get_random_connection(enabled=True)

        # getting a random enabled connection may be impossible if all conections are disabled
        if split_src_layer_id is None:
            return

        split_src_layer, split_dst_layer = self.layers[split_src_layer_id], self.layers[split_dst_layer_id]

        # find existing candidate layers for new node
        node_layers = [layer for layer in self.layers if layer["src"] == split_src_layer_id and layer["dst"] == split_dst_layer_id]
        
        if len(node_layers) == 0:

            node_layer = self.add_layer(split_src_layer, split_dst_layer)

            # new layer may take up too much RAM
            if node_layer is None:
                return 
        else:            
            node_layer = np.random.choice(node_layers)

        new_node_ix = split_src_node * split_dst_layer["size"] + split_dst_node

        # disable split connection
        dst_in_src_startix = self.get_startix(split_dst_layer, split_src_layer)
        self.connection_tables[split_src_layer["table_ix"]][split_src_node, dst_in_src_startix + split_dst_node] = False

        # enable lower connection from src layer to node layer
        nde_in_src_startix = self.get_startix(node_layer, split_src_layer)
        self.connection_tables[split_src_layer["table_ix"]][split_src_node, nde_in_src_startix + new_node_ix] = True

        # enable upper connection from node layer to dst layer
        dst_in_nde_startix = self.get_startix(split_dst_layer, node_layer)
        self.connection_tables[node_layer["table_ix"]][new_node_ix, dst_in_nde_startix + split_dst_node] = True 
           

    def add_connection(self):
        """ Enables a random disabled connection. """

        genome = self.get_genome()
        disabled_ics = np.where(genome == False)[0]

        # enabling a disabled connection is impossible if there is no disabled connection
        if len(disabled_ics) == 0:
            print("Warning: No disabled connections found, so add_connection is impossible; skipping operation.")
            return

        genome[np.random.choice(disabled_ics)] = True
        self.set_genome(genome)
        

    def change_activation(self):
        """ Changes a random activation function. """
        
        self.activations[np.random.randint(0, self.activations.shape[0])] = np.random.choice(list(ACTIVATION_DICT.keys()))


    def get_activations(self):
        """ Returns activations. """

        return np.array(self.activations)


    def set_activations(self, activations):
        """ Sets activations. """
        
        assert activations.shape == self.activations.shape, "Incompatible activation sizes!"

        self.activations = activations


    def get_genome(self):
        """ Constructs genome of individuum from boolean connection tables. """

        return np.concatenate([table.flatten() for table in self.connection_tables])


    def set_genome(self, genome):
        """ Sets boolean connection tables from genome. """

        assert genome.size == self.get_genome().size, "Incompatible genome sizes! Use adapt_structure_to before copying genome."

        current_genix = 0

        for ix, table in enumerate(self.connection_tables): 

            self.connection_tables[ix] = genome[current_genix: current_genix + table.size].reshape(*table.shape)        
            current_genix += table.size


    def get_complexity(self):
        """ Returns number of enabled connections and number of potential connections as complexity metrics. """
        
        genome = self.get_genome()
        return genome.sum(), len(self.layers) - 2              


    def add_layer(self, src_layer, dst_layer):
        """ Adds a node layer slitting connections between a source and a destination layer. """

        new_layer = {
            "id": len(self.layers),
            "table_ix": len(self.connection_tables), 
            "size": src_layer["size"] * dst_layer["size"],
            "src": src_layer["id"],
            "dst": dst_layer["id"],
            "pos": (src_layer["pos"] + dst_layer["pos"]) / 2
        }

        # pop_size * len(self.connection_tables) * src_layer["size"] * 500_000 * 8 bit per boolean is about all 16 GB of RAM can sustain in worst case
        if new_layer["size"] > 500_000:
            print("Warning: Adding new layer would take up too much RAM; skipping operation.")
            return None

        self.layers.append(new_layer)
        self.rearrange_layers()

        new_layer_table_connections = []

        for ix, layer in enumerate(self.layers):

            # add connections to new layer from all lower layers
            if layer["pos"] < self.layers[-1]["pos"]:

                new_layer_connections = np.zeros((layer["size"], self.layers[-1]["size"]), dtype=bool)
                self.connection_tables[layer["table_ix"]] = np.concatenate([self.connection_tables[layer["table_ix"]], new_layer_connections], axis=1)

            # add connections from new layer to all upper layers
            elif layer["pos"] > self.layers[-1]["pos"]:

                new_layer_connections = np.zeros((self.layers[-1]["size"], layer["size"]), dtype=bool)
                new_layer_table_connections.append(new_layer_connections)
                
        # make new connection table with connections to all upper layers (sorted by id)
        new_layer_table = np.concatenate(new_layer_table_connections, axis=1)
        self.connection_tables.append(new_layer_table)
    
        # activations follow the same ordering as id (axis 1 of input_layer)
        self.activations = np.concatenate([self.activations, np.ones(new_layer["size"])], axis=0)

        return self.layers[-1]


    def delete_layer(self, del_layer_id):
        """ Deletes a node layer. """

        del_layer = self.layers[del_layer_id]

        self.connection_tables.pop(del_layer["table_ix"])
        self.layers.pop(del_layer_id)

        # adapt upper ids and table_ix to reflect new id
        for ix, layer in enumerate(self.layers[del_layer_id:]):
            layer["id"] -= 1
            layer["table_ix"] -= 1 

        self.rearrange_layers()
        
        # delete connections to del layer from all lower layers
        for ix, layer in enumerate(self.layers):
            if layer["pos"] < del_layer["pos"]:
                layer_table = self.connection_tables[layer["table_ix"]]
                del_layer_startix = self.get_startix(del_layer, layer)
                self.connection_tables[layer["table_ix"]] = np.delete(layer_table, list(range(del_layer_startix, del_layer_startix + del_layer["size"])), axis=1)

        # delete corresponding activations (following input layer axis 1 ordering)
        del_layer_input_startix = self.get_startix(del_layer, self.layers[0])
        self.activations = np.delete(self.activations, list(range(del_layer_input_startix, del_layer_input_startix + del_layer["size"])), axis=0)


    def adapt_structure_to(self, other):
        """ Adapts layer structure (and genome size) to the layer structure of another individuum. """
        
        assert self.layers[0]["size"] == other.layers[0]["size"], "Input layer size unequal!"
        assert self.layers[1]["size"] == other.layers[1]["size"], "Output layer size unequal!"

        other_pos_sortics = np.argsort([layer["pos"] for layer in other.layers])

        other_equivalences = {}
        lowest_correct_own_pos = 0
        other_layers_to_be_replicated = []

        for other_pos, other_layer_ix in enumerate(other_pos_sortics):

            other_layer = other.layers[other_layer_ix]
            own_pos_sortics = np.argsort([layer["pos"] for layer in self.layers])

            for own_pos, own_layer_ix in enumerate(own_pos_sortics):

                own_layer = self.layers[own_layer_ix] 

                # assume layers to be equivalent if size is equal
                if own_layer["size"] == other_layer["size"]:

                    lowest_correct_own_pos = own_pos
                    other_equivalences[other_layer_ix] = own_layer

                    # delete lower layers if higher position (= unnecessary layers in between)
                    if own_pos > other_pos:
                        
                        delete_layer_ids = own_pos_sortics[lowest_correct_own_pos + 1: own_pos]

                        for layer_id in np.sort(delete_layer_ids)[::-1]:
                            self.delete_layer(layer_id)

                    break
            else:
                
                # not equivalent to any own layer, must be replicated later
                other_layers_to_be_replicated.append(other_layer_ix)
                
        # delete leftover non-equivalent own layers
        leftover_ids = [layer["id"] for layer in self.layers if layer not in list(other_equivalences.values())]

        for layer_id in leftover_ids:
            self.delete_layer(layer_id)

        # replicate other layers from list by ascending id (same order they were created in other) to avoid dependency issues
        for other_layer_id in np.sort(other_layers_to_be_replicated):

            own_src_layer = other_equivalences[other.layers[other_layer_id]["src"]]
            own_dst_layer = other_equivalences[other.layers[other_layer_id]["dst"]]

            other_equivalences[other_layer_id] = self.add_layer(own_src_layer, own_dst_layer)


    def predict(self, inputs, weights):
        """ Assigns weight to all enabled connections and predits outputs for all inputs. """
    
        assert inputs.shape[1] == self.layers[0]["size"], "Input samples have wrong dimensionality!"

        # order layers by position
        pos_ordered_layers = [self.layers[ix] for ix in np.argsort([layer["pos"] for layer in self.layers])]

        # determine hidden layers
        hidden_layers = pos_ordered_layers[1:-1]
        hidden_nodes = [np.empty(layer["size"], dtype=np.float32) for layer in hidden_layers]
 
        # make lookup dictionaries for startics from each layer to each hidden layer (input can never be destination, output startx is always 0)
        startics = [{layer["id"]: self.get_startix(h_layer, layer) for layer in self.layers if layer["pos"] < h_layer["pos"]} for h_layer in hidden_layers]

        # make tables real-values (multiplication with single shared or multiple weights)
        if type(weights) == np.float64:

            input_connection_table = self.connection_tables[0] * weights
            hidden_tables = [self.connection_tables[layer["table_ix"]] * weights for layer in hidden_layers]

        elif type(weights) == np.ndarray:

            assert weights.size == self.get_genome().size, "Wrong weight size!"
        
            input_connection_table = weights[:self.connection_tables[0].size].reshape(*self.connection_tables[0].shape) * self.connection_tables[0]
            current_weightix = self.connection_tables[0].size

            hidden_tables = []

            for layer in hidden_layers:

                table = self.connection_tables[layer["table_ix"]]
                hidden_tables.append(weights[current_weightix : current_weightix + table.size].reshape(*table.shape) * table)
                current_weightix += table.size

        else:
            raise Exception("Given weights are of unexpected type.")

        output_layer = self.layers[1]
        batch_inputs = inputs.T
        
        # for all hidden layers
        for hidden_ix, (hidden_layer, startics_lookup) in enumerate(zip(hidden_layers, startics)):

            # gather hidden drive from input nodes
            input_startix = startics_lookup[0]
            input_weight_matrix = input_connection_table[:, input_startix : input_startix + hidden_layer["size"]].T
            hidden_nodes[hidden_ix] = np.dot(input_weight_matrix, batch_inputs)

            # add drives from previous hidden layers
            for pre_hidden_ix, pre_hidden_layer in enumerate(hidden_layers[:hidden_ix]):
                hidden_startix = startics_lookup[pre_hidden_layer["id"]]
                pre_hidden_weight_matrix = hidden_tables[pre_hidden_ix][:, hidden_startix : hidden_startix + hidden_layer["size"]].T
                hidden_nodes[hidden_ix] += np.dot(pre_hidden_weight_matrix, hidden_nodes[pre_hidden_ix])

            # apply corresponding activation function to each node
            for node_ix, node_val in enumerate(hidden_nodes[hidden_ix]):
                activation_id = self.activations[input_startix + node_ix]
                hidden_nodes[hidden_ix][node_ix] = ACTIVATION_DICT[activation_id](node_val)

        # gather output drive from input nodes
        input_weight_matrix = input_connection_table[:, :output_layer["size"]].T
        output_nodes = np.dot(input_weight_matrix, batch_inputs)

        # add drives from hidden layers
        for hidden_ix, hidden_layer in enumerate(hidden_layers):
            hidden_weight_matrix = hidden_tables[hidden_ix][:, :output_layer["size"]].T
            output_nodes += np.dot(hidden_weight_matrix, hidden_nodes[hidden_ix])

        # apply corresponding activation function to each node
        for node_ix, node_val in enumerate(output_nodes):
            activation_id = self.activations[node_ix]
            output_nodes[node_ix] = ACTIVATION_DICT[activation_id](node_val)

        return output_nodes.T


    def get_random_connection(self, enabled=True):
        """ Returns source and destination layer and node indices of a random connection. """

        relevant_connections = []

        for ix, table_layer in enumerate(self.layers):

            # no connection table for output layer
            if ix != 1:

                # src layer index contains only table layer
                src_layer_index = np.array(table_layer["size"] * [table_layer["id"]])

                # dst layer index contains all layers with higher position in id (=creation) order
                dst_layer_index = np.array([c for layer in self.layers if layer["pos"] > table_layer["pos"] for c in layer["size"] * [layer["id"]]])

                if enabled:
                    src_ics, dst_ics = np.where(self.connection_tables[table_layer["table_ix"]] == True)
                else:
                    src_ics, dst_ics = np.where(self.connection_tables[table_layer["table_ix"]] == False)
                    
                src_layer_ids = src_layer_index[src_ics]
                dst_layer_ids = dst_layer_index[dst_ics]

                # subtract layer-specific startix
                dst_startics = {dst_layer_id: self.get_startix(self.layers[dst_layer_id], table_layer) for dst_layer_id in np.unique(dst_layer_ids)}
                dst_ics = [dst_ix - dst_startics[dst_layer_id] for dst_ix, dst_layer_id in zip(dst_ics, dst_layer_ids)]

                relevant_connections.extend(list(zip(src_layer_ids, src_ics, dst_layer_ids, dst_ics)))

        # there may be no relevant connections
        if len(relevant_connections) == 0:
            print("Warning: No relevant connections to choose random connection from; skipping operation.")
            return (None, None, None, None)
        else:
            return relevant_connections[np.random.randint(0, len(relevant_connections))]


    def rearrange_layers(self):
        """ Re-arranges positions of layers. """
            
        layer_positions = [layer["pos"] for layer in self.layers]
        layer_pos_ranked = np.argsort(np.argsort(layer_positions))

        for layer, ranked_pos in zip(self.layers, layer_pos_ranked):
            layer["pos"] = ranked_pos



    def get_startix(self, destination_layer, source_layer):
        """ Returns the starting column index of a destination node layer in the connection table destination index (axis 1) of a source node layer. """

        return sum([layer["size"] for layer in self.layers[:destination_layer["id"]] if layer["pos"] > source_layer["pos"]])


