# import numpy as np
# from matplotlib import pyplot as plt

# class Izhikevich:
#     def __init__(self, a, b, c, d, Vth, T, dt):
#         self.a = a
#         self.b = b
#         self.c = c
#         self.d = d
#         self.Vth = Vth
#         self.u = self.b * self.c
#         self.T = T
#         self.dt = dt
#         self.t = np.arange(0, self.T, self.dt)
#         self.in_synapes = []
#         self.out_synapes = []

#     # I is an array of length self.t
#     def run(self, I):
#         V = np.zeros(len(self.t))
#         V[0] = self.c
#         u = np.zeros(len(self.t))
#         u[0] = self.u
#         num_spikes = 0
#         for t in range(1, len(self.t)):
#             dv = ((0.04 * V[t - 1] ** 2) + (5 * V[t - 1]) + 140 - u[t - 1] + I[t - 1]) * self.dt
#             du = (self.a * ((self.b * V[t - 1]) - u[t - 1])) * self.dt
#             V[t] = V[t - 1] + dv
#             u[t] = u[t - 1] + du

#             if V[t] >= self.Vth:
#                 V[t] = self.c
#                 u[t] = self.d + u[t]
#                 num_spikes += 1

#         return V, num_spikes

# T = 1000
# dt = 0.1
# t = np.arange(0, T, dt)
# izhi = Izhikevich(0.02, 0.2, -65, 2, 30, T, dt)

# I = 25
# i_inj = np.linspace(0, I, t.shape[0])
# # i_inj = np.full(t.shape, I)
# V, num_spikes = izhi.run(i_inj)

# print(num_spikes)

# plt.figure()
# plt.subplot(2,1,1)
# plt.title('Izhi Neuron')
# plt.plot(t, V, 'k')
# plt.ylabel('V')

# plt.subplot(2,1,2)
# plt.plot(t, i_inj, 'k')
# plt.xlabel('t')
# plt.ylabel('$I_{inj}$ ($\\mu{A}/cm^2$)')

# plt.show()


# class Synapse:
#     def __init__(self):
#         self.weight = 1

#     def hebbian(self):
#         self.weight = self.weight + 0.1

#     def anti_hebbian(self):
#         self.weight = max(0, self.weight - 0.1)

# # The snn network!
# class Network:
#     def __init__(self, a, b, c, d, Vth, T, dt):
#         self.layer_1 = [Izhikevich(a, b, c, d, Vth, T, dt) for i in range(200)]
#         self.synapse_layer = [[Synapse() for j in range(200)] for i in range(10)]
#         self.layer_2 = [Izhikevich(a, b, c, d, Vth, T, dt) for i in range(10)]

#         for i, synapse1_ in enumerate(self.synapse_layer):
#             for j, synapse_ in enumerate(synapse1_):
#                 self.layer_2[i].in_synapes.append(synapse_)
#                 self.layer_1[j].out_synapes.append(synapse_)

#     """
#     1. Receive event-encoded audio data as input to the network. This data is assumed to be in a format compatible with the neurons' input requirements, i.e., a time series of current injections for each neuron in the first layer.
#     2. Process the input through the first layer of neurons, simulating their spiking behavior over time.
#     3. Transmit spikes to the second layer through synapses, with each synapse potentially modulating the spikes based on its weight.
#     4. Process the input in the second layer of neurons, again simulating their spiking behavior.
#     """
#     def forward(self, event_img):
#         # Assume event_img shape is (batch_size, timesteps, features)
#         print("event image: ", event_img)
#         print("event image: ", event_img.shape)

#         batch_size, timesteps, features = event_img.shape
        
#         output_spikes = np.zeros((batch_size, len(self.layer_2)))

#         for sample_index in range(batch_size):
#             print("sample index: ",sample_index)
#             input_currents = np.sum(event_img[sample_index], axis=0)
#             layer_1_outputs = [neuron.run(np.full((len(neuron.t),), input_currents[i]))[1] for i, neuron in enumerate(self.layer_1)]
            
#             # Placeholder for simulated layer 2 inputs for learning rule application
#             synapse_inputs = np.zeros((len(self.layer_2),))

#             # Adjust synaptic weights based on a simple Hebbian/anti-Hebbian principle before calculating layer 2 inputs
#             for i, output in enumerate(layer_1_outputs):
#                 for j, synapse in enumerate(self.layer_1[i].out_synapes):
#                     # For this example, assume a simple condition for learning:
#                     # If the layer 1 neuron fires, we consider it for Hebbian learning.
#                     if output > 0:
#                         synapse.hebbian()  # Strengthen the synapse
#                     else:
#                         synapse.anti_hebbian()  # Weaken the synapse
                    
#                     # Calculate the input to layer 2 neurons as sum of weighted spikes
#                     synapse_inputs[j] += output * synapse.weight

#             # Process the adjusted inputs in layer 2 neurons
#             for i, neuron in enumerate(self.layer_2):
#                 output_spikes[sample_index, i] = neuron.run(np.full((len(neuron.t),), synapse_inputs[i]))[1]

#         return output_spikes


# class NeuronNetwork:
#     def __init__(self, a, b, c, d, Vth, T, dt):
#         self.layer_1 = [Izhikevich(a, b, c, d, Vth, T, dt) for i in range(200)]
#         self.synapse_layer = [[Synapse() for j in range(200)] for i in range(10)]
#         self.layer_2 = [Izhikevich(a, b, c, d, Vth, T, dt) for i in range(10)]



import numpy as np

class Izhikevich:
    """Izhikevich neuron model"""
    def __init__(self, a, b, c, d, Vth, T, dt):
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.Vth = Vth
        self.u = self.b * self.c
        self.T = T
        self.dt = dt
        self.t = np.arange(0, self.T, self.dt)
        self.I = 0

    def run(self, I):
        self.I = I
        V = np.zeros(len(self.t))
        V[0] = self.c
        u = np.zeros(len(self.t))
        u[0] = self.u
        num_spikes = 0
        for t in range(1, len(self.t)):
            dv = ((0.04 * V[t - 1] ** 2) + (5 * V[t - 1]) + 140 - u[t - 1] + self.I) * self.dt
            du = (self.a * ((self.b * V[t - 1]) - u[t - 1])) * self.dt
            V[t] = V[t - 1] + dv
            u[t] = u[t - 1] + du

            if V[t] >= self.Vth:
                V[t] = self.c
                u[t] = u[t - 1] + self.d
                num_spikes += 1
        return V, num_spikes

# weights
class Synapse:
    def __init__(self, pre, post, initial_weight=1.0):
        self.pre = pre
        self.post = post
        self.weight = initial_weight

    def hebbian(self):
        self.weight += 0.1

    def anti_hebbian(self):
        self.weight = max(0, self.weight - 0.1)


class Network:
    def __init__(self, a, b, c, d, Vth, T, dt, input_dimension, output_dimension):
        self.input_neurons = [Izhikevich(a, b, c, d, Vth, T, dt) for _ in range(input_dimension)]
        self.output_neurons = [Izhikevich(a, b, c, d, Vth, T, dt) for _ in range(output_dimension)]
        
        # synapses connecting input neurons to output neurons
        self.synapses_ih = []
        for pre_idx in range(input_dimension):
            for post_idx in range(output_dimension):
                # Directly store references to neuron objects in Synapse instances
                synapse = Synapse(pre_idx, self.output_neurons[post_idx])
                self.synapses_ih.append(synapse)

    def forward(self, input_spikes):
        # input_spikes.shape == (batches, num_time_steps, num_input_neurons)
        batch_size, num_time_steps, _ = input_spikes.shape
        
        output_spikes = np.zeros((batch_size, len(self.output_neurons)))
        
        for example_index in range(batch_size):
            # Reset the input current ( to be reviewed )
            for neuron in self.output_neurons:
                neuron.I = 0


            # Store spikes for Hebbian/ anti hebbian ( to be reviewed )
            input_spikes_sum = np.zeros(len(self.input_neurons))
            output_spikes_sum = np.zeros(len(self.output_neurons))

            for time_step in range(num_time_steps):
                current_input = input_spikes[example_index, time_step, :]
                
                # Apply the current input to the input neurons, and then propagate to output neurons
                for synapse in self.synapses_ih:
                    input_spike = current_input[synapse.pre]  # Spike from the current input neuron

                    input_spikes_sum[synapse.pre] += input_spike
                    
                    # Now synapse.post directly references an output neuron object
                    synapse.post.I += input_spike * synapse.weight  # Weighted input to the output neuron

            
            # Calculate the response of output neurons (after time steps)
            for i, neuron in enumerate(self.output_neurons):
                _, num_spikes = neuron.run(neuron.I)  # check later on what parameter to set it to!
                
                # Track spikes for each output neuron
                output_spikes_sum[i] = num_spikes  
                output_spikes[example_index, i] = num_spikes
            
            # Hebbian or Anti-Hebbian learning
                    

        return output_spikes