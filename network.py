
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
    def __init__(self, pre, post, initial_weight):
        self.pre = pre
        self.post = post
        self.weight = initial_weight

    def hebbian(self):
        self.weight += 0.1

    def anti_hebbian(self):
        self.weight = max(0, self.weight - 0.1)


# class Network:
#     def __init__(self, a, b, c, d, Vth, T, dt, input_dimension, output_dimension):
#         self.input_neurons = [Izhikevich(a, b, c, d, Vth, T, dt) for _ in range(input_dimension)]
#         self.output_neurons = [Izhikevich(a, b, c, d, Vth, T, dt) for _ in range(output_dimension)]

#         self.synapses_ih = [[Synapse(self.input_neurons[i], self.output_neurons[j], 0.5) 
#                              for j in range(output_dimension)] for i in range(input_dimension)]


#     def forward(self, input_data, output_label):
#         output_spikes = np.zeros((len(self.output_neurons)))

#         # forward pass
#         for j in range(len(self.output_neurons)):
#             weighted_input_sum = 0
#             for i in range(len(self.input_neurons)):
#                 synapse = self.synapses_ih[i * len(self.output_neurons) + j]
#                 weighted_input_sum += self.input_neurons[i].run(input_data[i])[1] * synapse.weight
#             output_spikes[j] = weighted_input_sum
            

#         predicted_index = np.argmax(output_spikes[:])

#         print(output_spikes)

#         # learning: adjust based on the output and expected output
#         for i in range(len(self.input_neurons)):
#             for j in range(len(self.output_neurons)):
#                 synapse = self.synapses_ih[i * len(self.output_neurons) + j]

#                 # hebbian if the predicted neuron is the same as the label
#                 if j == predicted_index and input_data[i] > 0: # did neuron actually fire???
#                     if predicted_index == output_label:
#                         synapse.hebbian()
#                     else:  # Incorrect prediction
#                         synapse.anti_hebbian() 


#         return output_spikes
        

class Network:
    def __init__(self, a, b, c, d, Vth, T, dt, input_dimension, output_dimension):
        self.input_neurons = [Izhikevich(a, b, c, d, Vth, T, dt) for _ in range(input_dimension)]
        self.output_neurons = [Izhikevich(a, b, c, d, Vth, T, dt) for _ in range(output_dimension)]

        self.synapses_ih = [[Synapse(self.input_neurons[i], self.output_neurons[j], 0.5) 
                             for j in range(output_dimension)] for i in range(input_dimension)]


    def forward(self, input_data, output_label):
        output_spikes = np.zeros(len(self.output_neurons))

        # Forward pass through the network
        for j in range(len(self.output_neurons)):
            weighted_input_sum = 0
            for i in range(len(self.input_neurons)):
                synapse = self.synapses_ih[i][j]
                neuron_output, _ = self.input_neurons[i].run(input_data[i])
                weighted_input_sum += neuron_output * synapse.weight
            output_spikes[j] = weighted_input_sum

        predicted_index = np.argmax(output_spikes)

        # Learning process based on prediction accuracy
        for i in range(len(self.input_neurons)):
            for j in range(len(self.output_neurons)):
                synapse = self.synapses_ih[i][j]
                if j == predicted_index and input_data[i] > 0:
                    if predicted_index == output_label:
                        synapse.hebbian()
                    else:
                        synapse.anti_hebbian()

        return output_spikes
