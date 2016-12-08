#include <iostream>
#include <cmath>
#include <stdio.h>
#include <random>

// define global
int const rows = 10;
int const columns = 10;

template <typename T>
struct neuron_constants
{
    T i_syn_0;
    T e_l;
    T tau_l;
    T tau_syn;
    T threshold;
    T reset_potential;
    T refrac_period;
};


struct neuron_state_var
{
    double voltage;
    double syn_current;
    int fired;
    int refrac;
    int refrac_count;
    neuron_constants<double> constants;
};


struct neural_net
{
    double weigths[rows][columns];
    neuron_state_var neurons[columns];
};

void read_in_config(neural_net *nn)
{
    FILE *fr;
    char weigth_in[] = "weigth_config.data";
    fr = fopen(weigth_in, "r");
    for (int i_row=0; i_row<rows; i_row++)
    {
        for (int j_column=0; j_column<columns; j_column++)
        {
            fscanf(fr, "%lf ", &nn->weigths[i_row][j_column]);
        }
    }
    fclose(fr);

    char neuron_config_in[] = "neuron_config.data";
    fr = fopen(neuron_config_in, "r");

    for (int j_column = 0; j_column<columns; j_column++)
    {
        fscanf(fr, "%lf ", &nn->neurons[j_column].constants.e_l);
        fscanf(fr, "%lf ", &nn->neurons[j_column].constants.i_syn_0);
        fscanf(fr, "%lf ", &nn->neurons[j_column].constants.tau_l);
        fscanf(fr, "%lf ", &nn->neurons[j_column].constants.tau_syn);
        fscanf(fr, "%lf ", &nn->neurons[j_column].constants.reset_potential);
        fscanf(fr, "%lf ", &nn->neurons[j_column].constants.threshold);
        fscanf(fr, "%lf ", &nn->neurons[j_column].constants.refrac_period);
    }
    fclose(fr);
}


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
    } else {
        return_value = 0;
    }
    return return_value;
}

double synaptic_input_right_side(double syn_current_input, neuron_constants<double> constants, int step, int *spiketimes, double weigthsum)
{
    return -1.0/constants.tau_syn * syn_current_input + constants.i_syn_0 * spiketimes[step] + constants.i_syn_0 * weigthsum;
}


double lif_right_side(double v, neuron_constants<double> constants, int step, double (*current_input)(int), double syn_current_input)
{
    return 1.0/constants.tau_l * (constants.e_l - v) + current_input(step) + syn_current_input;
}

neuron_state_var explicit_euler_step(neuron_state_var neuron_state, neuron_constants<double> constants, const double d_t, int step, double (*current_input)(int), int *spiketimes, double weigthsum)
{
    double k_1_s = synaptic_input_right_side(neuron_state.syn_current, constants, step, spiketimes, weigthsum);
    neuron_state.syn_current = neuron_state.syn_current + k_1_s * d_t;

    double k_1_n = lif_right_side(neuron_state.voltage, constants, step, current_input, neuron_state.syn_current);
    neuron_state.voltage = neuron_state.voltage + k_1_n * d_t;

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


neuron_state_var runge_kutta_step(neuron_state_var neuron_state, neuron_constants<double> constants, const double d_t, int step, double (*current_input)(int), int *spiketimes, double weigthsum)
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
        if (neuron_state.refrac_count > constants.refrac_period / d_t)
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
        {
            weight_sum += nn.weigths[neuron][i];
        }
    }
    return weight_sum;
}

int rand_val_0_1(double p)
{
    double rand_double = (double)rand() / RAND_MAX;
    return rand_double > (1 - p);
}

void init_rdm_spike_train(int num_sim_steps, int *spiketimes, double p)
{
    for(int i=0; i<num_sim_steps; i++)
    {
        spiketimes[i] = rand_val_0_1(p);
    }
}

void evolve_net(neural_net nn, double t, double d_t, int num_sim_steps, int step, double (*current_input)(int), int *spiketimes)
{
    FILE *fp;
    char output[] = "simulator_output.sim_data";
    fp = fopen(output,"w");

    double weigthsum;
    for (int i=0; i<num_sim_steps; i++)
    {
        for (int j=0; j < columns; j++){
            weigthsum = sum_weigths_in_timestep(nn, j);
            nn.neurons[j] = runge_kutta_step(nn.neurons[j], nn.neurons[j].constants, d_t, step, current_input, spiketimes, weigthsum);
            fprintf(fp, "%f ", nn.neurons[j].voltage);
            fprintf(fp, "%f ", nn.neurons[j].syn_current);
            fprintf(fp, "%d ", nn.neurons[j].fired);
        }
        t += d_t;
        step += 1;
        fprintf(fp, "%f\n", t);
    }
    fclose(fp);
}

void neuron_config_hom_setup(neural_net *nn)
{
    for (int i=0; i<columns; i++)
    {
       nn->neurons[i].constants.e_l = 0.0;
       nn->neurons[i].constants.i_syn_0 = 4000.0;
       nn->neurons[i].constants.tau_l = 0.01;
       nn->neurons[i].constants.tau_syn = 3.0;
       nn->neurons[i].constants.reset_potential= 0.0;
       nn->neurons[i].constants.threshold = 1.2;
       nn->neurons[i].constants.refrac_period = 1;
    }
}

void weigth_config_non_self_all_all(neural_net *nn)
{
    for(int i=0; i<columns; i++)
    {
        for(int j=0; j<rows; j++)
        {
            if (i != j)
                nn->weigths[i][j] = 0.0001;
            if (i == j)
                nn->weigths[i][j] = 0;
        }
    }
}

void set_initial_conditions(neural_net *nn)
{
    for (int i=0; i<columns; i++)
    {
        nn->neurons[i].voltage = 0.0;
        nn->neurons[i].syn_current = 0.0;
        nn->neurons[i].refrac = 0;
        nn->neurons[i].refrac_count = 0;
    }
}

void print_config(neural_net *nn)
{
    printf("\nPrinting the weigth matrix \n");
    printf("-----------------------------\n");
    for (int i=0; i<rows; i++)
    {
        for (int j=0; j<columns; j++)
        {
            printf("%f ", nn->weigths[i][j]);
            if (j == columns - 1)
                printf("\n");
        }
    }
    printf("\nPrint config for neuron 0 \n");
    printf("-----------------------------\n");
    printf("e_l:           %f \n", nn->neurons[0].constants.e_l = 0.0);
    printf("i_syn:         %f \n", nn->neurons[0].constants.i_syn_0 = 4000.0);
    printf("tau_l          %f \n", nn->neurons[0].constants.tau_l = 0.01);
    printf("tau_syn        %f \n", nn->neurons[0].constants.tau_syn = 3.0);
    printf("reset_potential%f \n", nn->neurons[0].constants.reset_potential= 0.0);
    printf("threshold:     %f \n", nn->neurons[0].constants.threshold = 1.2);
    printf("refrac_period: %f \n", nn->neurons[0].constants.refrac_period = 1);
}

int main()
{
    int num_sim_steps = 200000;
    int step = 0;

    // init the neural net
    neural_net nn;

    // initializing the constants
    //neuron_config_hom_setup(&nn);
    //weigth_config_non_self_all_all(&nn);
    read_in_config(&nn);
    set_initial_conditions(&nn);
    //print_config(&nn);
    double d_t = 0.1e-3;  // count time in s

    // initial conditions
    double t = 0.0;
    int spiketimes[num_sim_steps];
    init_rdm_spike_train(num_sim_steps, spiketimes, 1500.0 * d_t); // average of 10 spikes per 1s

    evolve_net(nn, t, d_t, num_sim_steps, step, external_current, spiketimes);
}
