#include <iostream>
#include <cmath>
#include <stdio.h>
#include <random>

// define global
int const rows = 10;
int const columns = 10;

struct neuron_constants
{
    double i_syn_0;
    double e_l;
    double tau_l;
    double tau_syn;
    double threshold;
    double reset_potential;
    double refrac_period;
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


class Neural_net_c
{
    neural_net nn;
    double t;
    double d_t;
    int num_sim_steps;
    int step;
    int spiketimes[200000];


    public:
    void read_in_config();
    double external_current(int);
    double synaptic_input_right_side(double, double, int, int*, int);
    double lif_right_side(double, int, int);
    neuron_state_var explicit_euler_step(double, int, int*, int, double);
    neuron_state_var runge_kutta_step(double, int, int*, int, double);
    double sum_weigths_in_timestep(int);
    int rand_val_0_1(double p);
    void init_rdm_spike_train(int, int*, double);
    void evolve_net();
    void neuron_config_hom_setup();
    void weifgth_config_non_self_all_all();
    void set_initial_conditions();
    void print_config();
    void init_simulator();
};


void Neural_net_c::read_in_config()
{
    FILE *fr;
    char weigth_in[] = "weigth_config.data";
    fr = fopen(weigth_in, "r");
    for (int i_row=0; i_row<rows; i_row++)
    {
        for (int j_column=0; j_column<columns; j_column++)
        {
            fscanf(fr, "%lf ", &this->nn.weigths[i_row][j_column]);
        }
    }
    fclose(fr);

    char neuron_config_in[] = "neuron_config.data";
    fr = fopen(neuron_config_in, "r");

    for (int j_column = 0; j_column<columns; j_column++)
    {
        fscanf(fr, "%lf ", &this->nn.neurons[j_column].constants.e_l);
        fscanf(fr, "%lf ", &this->nn.neurons[j_column].constants.i_syn_0);
        fscanf(fr, "%lf ", &this->nn.neurons[j_column].constants.tau_l);
        fscanf(fr, "%lf ", &this->nn.neurons[j_column].constants.tau_syn);
        fscanf(fr, "%lf ", &this->nn.neurons[j_column].constants.reset_potential);
        fscanf(fr, "%lf ", &this->nn.neurons[j_column].constants.threshold);
        fscanf(fr, "%lf ", &this->nn.neurons[j_column].constants.refrac_period);
    }
    fclose(fr);
}


double Neural_net_c::external_current(int step)
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

double Neural_net_c::synaptic_input_right_side(double current, double weigthsum, int n_i, int *spiketimes, int step)
{
    return -1.0/this->nn.neurons[n_i].constants.tau_syn * current+ this->nn.neurons[n_i].constants.i_syn_0 * spiketimes[step] + this->nn.neurons[n_i].constants.i_syn_0 * weigthsum;
}

double Neural_net_c::lif_right_side(double voltage, int n_i, int step)
{
    return 1.0/this->nn.neurons[n_i].constants.tau_l * (this->nn.neurons[n_i].constants.e_l - voltage) + external_current(step) + this->nn.neurons[n_i].syn_current;
}

neuron_state_var Neural_net_c::explicit_euler_step(double weigthsum, int n_i, int *spiketimes, int step, double d_t)
{
    double k_1_s = synaptic_input_right_side(this->nn.neurons[n_i].syn_current, weigthsum, n_i, spiketimes, step);
    this->nn.neurons[n_i].syn_current = this->nn.neurons[n_i].syn_current + k_1_s * d_t;

    double k_1_n = lif_right_side(this->nn.neurons[n_i].voltage, n_i, step);
    this->nn.neurons[n_i].voltage = this->nn.neurons[n_i].voltage + k_1_n * d_t;

    if (this->nn.neurons[n_i].refrac)
    {
        this->nn.neurons[n_i].voltage = this->nn.neurons[n_i].constants.reset_potential;
        this->nn.neurons[n_i].refrac_count += 1;
        if (this->nn.neurons[n_i].refrac_count > this->nn.neurons[n_i].constants.refrac_period)
            this->nn.neurons[n_i].refrac = 0;
    }

    if (this->nn.neurons[n_i].voltage > this->nn.neurons[n_i].constants.threshold)
    {
        this->nn.neurons[n_i].voltage = this->nn.neurons[n_i].constants.reset_potential;
        this->nn.neurons[n_i].fired = 1;
        this->nn.neurons[n_i].refrac = 1;
        this->nn.neurons[n_i].refrac_count = 0;
    }
    else
        this->nn.neurons[n_i].fired = 0;
    return this->nn.neurons[n_i];
}


neuron_state_var Neural_net_c::runge_kutta_step(double weigthsum, int n_i, int *spiketimes, int step, double d_t)
{
    double k_1_s = synaptic_input_right_side(this->nn.neurons[n_i].syn_current, weigthsum, n_i, spiketimes, step);
    double k_2_s = synaptic_input_right_side(this->nn.neurons[n_i].syn_current + k_1_s * d_t, weigthsum, n_i, spiketimes, step);

    this->nn.neurons[n_i].syn_current = this->nn.neurons[n_i].syn_current + (k_1_s + k_2_s) / 2.0 * d_t;

    double k_1_n = lif_right_side(this->nn.neurons[n_i].voltage, n_i, step);
    double k_2_n = lif_right_side(this->nn.neurons[n_i].voltage + k_1_n * d_t, n_i, step);

    this->nn.neurons[n_i].voltage = this->nn.neurons[n_i].voltage + (k_1_n + k_2_n) / 2.0 * d_t;

    if (this->nn.neurons[n_i].refrac)
    {
        this->nn.neurons[n_i].voltage = this->nn.neurons[n_i].constants.reset_potential;
        this->nn.neurons[n_i].refrac_count += 1;
        if (this->nn.neurons[n_i].refrac_count > this->nn.neurons[n_i].constants.refrac_period / d_t)
            this->nn.neurons[n_i].refrac = 0;
    }

    if (this->nn.neurons[n_i].voltage > this->nn.neurons[n_i].constants.threshold)
    {
        this->nn.neurons[n_i].voltage = this->nn.neurons[n_i].constants.reset_potential;
        this->nn.neurons[n_i].fired = 1;
        this->nn.neurons[n_i].refrac = 1;
        this->nn.neurons[n_i].refrac_count = 0;
    }
    else
        this->nn.neurons[n_i].fired = 0;
    return this->nn.neurons[n_i];
}

double Neural_net_c::sum_weigths_in_timestep(int neuron)
{
    double weight_sum = 0;
    for(int i=0; i<rows; i++)
    {
        if(this->nn.neurons[neuron].fired)
        {
            weight_sum += this->nn.weigths[neuron][i];
        }
    }
    return weight_sum;
}

int Neural_net_c::rand_val_0_1(double p)
{
    double rand_double = (double)rand() / RAND_MAX;
    return rand_double > (1 - p);
}

void Neural_net_c::init_rdm_spike_train(int num_sim_steps, int *spiketimes, double p)
{
    for(int i=0; i<num_sim_steps; i++)
    {
        spiketimes[i] = rand_val_0_1(p);
    }
}

void Neural_net_c::evolve_net()
{
    FILE *fp;
    char output[] = "simulator_output.sim_data";
    fp = fopen(output,"w");
    printf("%d \n", this->num_sim_steps);

    double weigthsum;
    for (int i=0; i<this->num_sim_steps; i++)
    {
        for (int j=0; j < columns; j++){
            weigthsum = sum_weigths_in_timestep(j);
            this->nn.neurons[j] = runge_kutta_step(weigthsum, j, this->spiketimes, this->step, this->d_t);
            fprintf(fp, "%f ", this->nn.neurons[j].voltage);
            fprintf(fp, "%f ", this->nn.neurons[j].syn_current);
            fprintf(fp, "%d ", this->nn.neurons[j].fired);
        }
        this->t += this->d_t;
        this->step += 1;
        fprintf(fp, "%f\n", this->t);
    }
    fclose(fp);
}

/*
void Neural_net_c::neuron_config_hom_setup()
{
    for (int i=0; i<columns; i++)
    {
       this->nn.neurons[i].constants.e_l = 0.0;
       this->nn.neurons[i].constants.i_syn_0 = 4000.0;
       this->nn.neurons[i].constants.tau_l = 0.01;
       this->nn.neurons[i].constants.tau_syn = 3.0;
       this->nn.neurons[i].constants.reset_potential= 0.0;
       this->nn.neurons[i].constants.threshold = 1.2;
       this->nn.neurons[i].constants.refrac_period = 1;
    }
}

void Neural_net_c::weigth_config_non_self_all_all()
{
    for(int i=0; i<columns; i++)
    {
        for(int j=0; j<rows; j++)
        {
            if (i != j)
                this->nn.weigths[i][j] = 0.0001;
            if (i == j)
                this->nn.weigths[i][j] = 0;
        }
    }
}
*/

void Neural_net_c::set_initial_conditions()
{
    for (int i=0; i<columns; i++)
    {
        this->nn.neurons[i].voltage = 0.0;
        this->nn.neurons[i].syn_current = 0.0;
        this->nn.neurons[i].refrac = 0;
        this->nn.neurons[i].refrac_count = 0;
    }
}

void Neural_net_c::print_config()
{
    printf("\nPrinting the weigth matrix \n");
    printf("-----------------------------\n");
    for (int i=0; i<rows; i++)
    {
        for (int j=0; j<columns; j++)
        {
            printf("%f ", this->nn.weigths[i][j]);
            if (j == columns - 1)
                printf("\n");
        }
    }
    printf("\nPrint config for neuron 0 \n");
    printf("-----------------------------\n");
    printf("e_l:           %f \n", this->nn.neurons[0].constants.e_l = 0.0);
    printf("i_syn:         %f \n", this->nn.neurons[0].constants.i_syn_0 = 4000.0);
    printf("tau_l          %f \n", this->nn.neurons[0].constants.tau_l = 0.01);
    printf("tau_syn        %f \n", this->nn.neurons[0].constants.tau_syn = 3.0);
    printf("reset_potential%f \n", this->nn.neurons[0].constants.reset_potential= 0.0);
    printf("threshold:     %f \n", this->nn.neurons[0].constants.threshold = 1.2);
    printf("refrac_period: %f \n", this->nn.neurons[0].constants.refrac_period = 1);
}

void Neural_net_c::init_simulator()
{
    this->num_sim_steps = 200000;
    this->step = 0;
    this->d_t = 0.1e-3;
    this->t = 0.0;
    init_rdm_spike_train(this->num_sim_steps, this->spiketimes, 1500.0 * this->d_t);
}

int main()
{
    Neural_net_c nn;
    nn.init_simulator();
    nn.set_initial_conditions();
    nn.read_in_config();
    
    nn.print_config();

    nn.evolve_net();
}
