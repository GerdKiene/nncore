#include <iostream>
#include <cmath>
#include <stdio.h>
#include <random>

// define global
int const rows = 1;
int const columns = 1;

struct neuron_constants
{
    double I_syn_0;
    double E_l;
    double tau_l;
    double tau_syn;
    double threshold;
    double reset_potential;
    int refrac_period;
};


struct neuron_state_var
{
    double voltage;
    double syn_current;
    int fired;
    int refrac;
    int refrac_count;
    neuron_constants constants;
};


struct neural_net
{
    double weigths[rows][columns];
    neuron_state_var neurons[columns];
};


double external_current(int step)
{
    double return_value;
    if(step < 500)
    {
        return_value = 0;
    }
    else if(step < 1000)
    {
        return_value = 0;
    }
    else
    {
        return_value = 0;
    }
    return return_value;
}

double synaptic_input_right_side(double syn_current_input, neuron_constants constants, int step, int *spiketimes, double weigthsum)
{
    return -1.0/constants.tau_syn * syn_current_input + constants.I_syn_0 * spiketimes[step] + constants.I_syn_0 * weigthsum;
}


double lif_right_side(double v, neuron_constants constants, int step, double (*current_input)(int), double syn_current_input)
{
    return 1.0/constants.tau_l * (constants.E_l - v) + current_input(step) + syn_current_input;
}

neuron_state_var runge_kutta_step(neuron_state_var neuron_state, neuron_constants constants, const double d_t, int step, double (*current_input)(int), int *spiketimes, double weigthsum)
{
    double k_1_s = synaptic_input_right_side(neuron_state.syn_current, constants, step, spiketimes, weigthsum);
    double k_2_s = synaptic_input_right_side(neuron_state.syn_current + k_1_s * d_t, constants, step, spiketimes, weigthsum);

    neuron_state.syn_current = neuron_state.syn_current + (k_1_s + k_2_s) / 2.0 * d_t;

    double k_1_n = lif_right_side(neuron_state.voltage, constants, step, current_input, neuron_state.syn_current);
    double k_2_n = lif_right_side(neuron_state.voltage + k_1_n * d_t, constants, step + 1, current_input, neuron_state.syn_current);

    neuron_state.voltage = neuron_state.voltage + (k_1_n + k_2_n) / 2.0 * d_t;

    if (neuron_state.refrac)
    {
        neuron_state.voltage = constants.reset_potential;
        neuron_state.refrac_count += 1;
        if (neuron_state.refrac_count > constants.refrac_period)
            neuron_state.refrac = 0;
    }

    if (neuron_state.voltage > constants.threshold)
    {
        neuron_state.voltage = constants.reset_potential;
        neuron_state.fired = 1;
        neuron_state.refrac = 1;
        neuron_state.refrac_count = 0;
    }
    else
        neuron_state.fired = 0;


    return neuron_state;
}

double sum_weigths_in_timestep(neural_net nn, int neuron)
{
    double weight_sum = 0;
    for(int i=0; i<rows; i++)
    {
        if(nn.neurons[neuron].fired)
            weight_sum += nn.weigths[neuron][i];
    }
    return weight_sum;
}

int rand_val_0_1(double p)
{
    double rand_double = (double)rand() / RAND_MAX;
    return rand_double > (1 - p);
}

void initialize_array(int num_sim_steps, int *spiketimes, double p)
{
    for(int i=0; i<num_sim_steps; i++)
    {
        spiketimes[i] = rand_val_0_1(p);
    }
}

void evolve_net(neural_net nn, double t, double d_t, int num_sim_steps, int step, double (*current_input)(int), int *spiketimes)
{

    FILE *fp;
    char output[] = "simulator_output";
    fp = fopen(output,"w");

    double weigthsum;
    for (int i=0; i<num_sim_steps; i++)
    {
        for (int j=0; j < columns; j++){
            weigthsum = sum_weigths_in_timestep(nn, j);
            nn.neurons[j] = runge_kutta_step(nn.neurons[j], nn.neurons[j].constants, d_t, step, current_input, spiketimes, weigthsum);
        }
        t += d_t;
        step += 1;
        fprintf(fp, "%f ", t);
        fprintf(fp, "%f ", nn.neurons[0].voltage);
        fprintf(fp, "%d ", nn.neurons[0].fired);
        fprintf(fp, "%f\n", nn.neurons[0].syn_current);
    }
    fclose(fp);
}


int main()
{

    int num_sim_steps = 2000;
    int step = 0;

    // init the neural net
    neural_net nn;

    // initializing the constants
    nn.neurons[0].constants.E_l = 1.0;
    nn.neurons[0].constants.I_syn_0 = 400.0;
    nn.neurons[0].constants.tau_l = 1.0;
    nn.neurons[0].constants.tau_syn = 3.0;
    nn.neurons[0].constants.reset_potential= 1.0;
    nn.neurons[0].constants.threshold = 2.9;
    nn.neurons[0].constants.refrac_period = 100;
    nn.weigths[0][0] = 0.5;
    double d_t = 0.01;

    // initial conditions
    nn.neurons[0].voltage = 1.0;
    nn.neurons[0].syn_current = 1.0;
    nn.neurons[0].refrac = 0;
    nn.neurons[0].refrac_count = 0;
    double t = 0.0;
    int spiketimes[num_sim_steps];
    initialize_array(num_sim_steps, spiketimes, 0.01 * d_t); // average of 10 spikes per 1s

    evolve_net(nn, t, d_t, num_sim_steps, step, external_current, spiketimes);
}
