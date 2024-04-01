
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
        V = np.zeros(len(self.t))
        V[0] = self.c
        u = np.zeros(len(self.t))
        u[0] = self.u
        num_spikes = 0
        I = np.zeros(len(self.t))
        I[0] = 0
        for t in range(1, len(self.t)):
            dv = ((0.04 * V[t - 1] ** 2) + (5 * V[t - 1]) + 140 - u[t - 1] + self.I[t-1]) * self.dt
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
        
        self.synapses_ih = [Synapse(i, output_neuron, np.random.uniform(0.5, 1.5)) for i in range(input_dimension) for output_neuron in self.output_neurons]
    

    def forward(self, input_spikes):
        batch_size, num_time_steps, num_input_neurons = input_spikes.shape
        print("input_spikes.shape: ", input_spikes.shape)
        output_spikes = np.zeros((batch_size, num_time_steps, len(self.output_neurons)))
        
        for example_index in range(batch_size):
            
            # Reset the input current for each example in the batch, eg. 1,2,3,...9
            for neuron in self.output_neurons: # 10 output neurons
                neuron.I = np.zeros(len(neuron.t))
            
            # Process each time step
            print("num_time_steps: ", num_time_steps)
            for time_step in range(num_time_steps):
                # selecting the input for all input neurons (:) at a specific time step (time_step) for a particular sample in the batch (example_index). 
                # effectively contains the input spikes that all input neurons receive at that moment
                current_input = input_spikes[example_index, time_step, :]
                print("current_input: ", current_input.shape) # (200,)
                
                # Apply the current input to the input neurons, and then propagate to output neurons
                print("self.synapses_ih: ", len(self.synapses_ih))
                for synapse in self.synapses_ih:
                    input_spike = current_input[synapse.pre]
                    # Accumulate input over time for each neuron
                    synapse.post.I += input_spike * synapse.weight  # adjust!
                
                # At each time step, update the state of each output neuron
                for i, neuron in enumerate(self.output_neurons):
                    V, num_spikes = neuron.run(neuron.I[time_step])
                    output_spikes[example_index, time_step, i] = num_spikes  # Recording spikes at each time step

                    # use V for learning?

            # learning (weight adjustment) AFTER processing all time steps
            
        # Learning, adjusting weights based on the temporal pattern of spikes
        
        return output_spikes