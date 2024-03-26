import numpy as np
from matplotlib import pyplot as plt

class Izhikevich:
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
        self.in_synapes = []
        self.out_synapes = []

    # I is an array of length self.t
    def run(self, I):
        V = np.zeros(len(self.t))
        V[0] = self.c
        u = np.zeros(len(self.t))
        u[0] = self.u
        num_spikes = 0
        for t in range(1, len(self.t)):
            dv = ((0.04 * V[t - 1] ** 2) + (5 * V[t - 1]) + 140 - u[t - 1] + I[t - 1]) * self.dt
            du = (self.a * ((self.b * V[t - 1]) - u[t - 1])) * self.dt
            V[t] = V[t - 1] + dv
            u[t] = u[t - 1] + du

            if V[t] >= self.Vth:
                V[t] = self.c
                u[t] = self.d + u[t]
                num_spikes += 1

        return V, num_spikes

T = 1000
dt = 0.1
t = np.arange(0, T, dt)
izhi = Izhikevich(0.02, 0.2, -65, 2, 30, T, dt)

I = 25
i_inj = np.linspace(0, I, t.shape[0])
# i_inj = np.full(t.shape, I)
V, num_spikes = izhi.run(i_inj)

print(num_spikes)

plt.figure()
plt.subplot(2,1,1)
plt.title('Izhi Neuron')
plt.plot(t, V, 'k')
plt.ylabel('V')

plt.subplot(2,1,2)
plt.plot(t, i_inj, 'k')
plt.xlabel('t')
plt.ylabel('$I_{inj}$ ($\\mu{A}/cm^2$)')

plt.show()


class Synapse:
    def __init__(self):
        self.weight = 1

    def hebbian(self):
        self.weight = self.weight + 0.1

    def anti_hebbian(self):
        self.weight = max(0, self.weight - 0.1)

# The snn network!
class Network:
    def __init__(self, a, b, c, d, Vth, T, dt):
        print("In!")
        self.layer_1 = [Izhikevich(a, b, c, d, Vth, T, dt) for i in range(200)]
        self.synapse_layer = [[Synapse() for j in range(200)] for i in range(10)]
        self.layer_2 = [Izhikevich(a, b, c, d, Vth, T, dt) for i in range(10)]

        for i, synapse1_ in enumerate(self.synapse_layer):
            for j, synapse_ in enumerate(synapse1_):
                self.layer_2[i].in_synapes.append(synapse_)
                self.layer_1[j].out_synapes.append(synapse_)

    """
    1. Receive event-encoded audio data as input to the network. This data is assumed to be in a format compatible with the neurons' input requirements, i.e., a time series of current injections for each neuron in the first layer.
    2. Process the input through the first layer of neurons, simulating their spiking behavior over time.
    3. Transmit spikes to the second layer through synapses, with each synapse potentially modulating the spikes based on its weight.
    4. Process the input in the second layer of neurons, again simulating their spiking behavior.
    """
    def forward(self, event_img):
        # Assume event_img shape is (batch_size, timesteps, features)
        print("event image shape inside forward: ", event_img.shape)
        batch_size, timesteps, features = event_img.shape
        
        # Output container for layer 2 spikes for each sample in the batch
        output_spikes = np.zeros((batch_size, len(self.layer_2)))
        
        for sample_index in range(batch_size):
            # Summarize spikes over timesteps for each neuron in layer 1
            input_currents = np.sum(event_img[sample_index], axis=0)  # Shape: (features,)
            
            # Run each neuron in layer 1 with the summarized input currents
            layer_1_outputs = [neuron.run(np.full((len(neuron.t),), input_currents[i]))[1] for i, neuron in enumerate(self.layer_1)]
            
            # Transmit spikes through synapses to layer 2
            synapse_outputs = np.zeros((len(self.layer_2),))
            for i, output in enumerate(layer_1_outputs):
                for j, synapse in enumerate(self.layer_1[i].out_synapes):
                    synapse_outputs[j] += output * synapse.weight  # Sum weighted spikes for each neuron in layer 2
            
            # Simulate layer 2 neurons with the received input
            for i, neuron in enumerate(self.layer_2):
                output_spikes[sample_index, i] = neuron.run(np.full((len(neuron.t),), synapse_outputs[i]))[1]
        
        return output_spikes

    
