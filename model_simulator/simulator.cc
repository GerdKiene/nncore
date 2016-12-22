#include <iostream>
#include <cmath>
#include <stdio.h>
#include <iostream>
#include <iomanip>
#include <random>
#include <fstream>
#include <sstream>
#include <string>
#include <typeinfo>

// define global
int const rows = 1;
int const columns = 1;

// saving the constants for individual neurons
template <typename T>
struct neuron_constants
{
    T i_syn_0;
    double i_syn_0_hidden;
    T e_l;
    double e_l_hidden;
    T tau_l;
    double tau_l_hidden;
    T one_over_tau_l;
    T tau_syn;
    double tau_syn_hidden;
    T one_over_tau_syn;
    T threshold;
    double threshold_hidden;
    T reset_potential;
    double reset_potential_hidden;
    T refrac_period;
    double refrac_period_hidden;
};

// state of an individual neuron, including the setup
template <typename T>
struct neuron_state_var
{
    T voltage;
    T syn_current;
    int fired;
    int refrac;
    int refrac_count;
    neuron_constants<T> constants;
};

// saves a neural net: a collection of neurons and thier connections
template <typename T>
struct neural_net
{
    T weigths[rows][columns];
    neuron_state_var<T> neurons[columns];
};


// class for simulating a neural net
template <class T>
class Neural_net_c
{
    T t;
    T d_t;
    int num_sim_steps;
    int step;
    int spiketimes[200000];


    public:
    neural_net<T> nn;
    void read_in_config();
    T external_current(int);
    T synaptic_input_right_side(T, T, int, int*, int);
    T lif_right_side(T, int, int);
    neuron_state_var<T> explicit_euler_step(T, int, int*, int, T);
    neuron_state_var<T> runge_kutta_step(T, int, int*, int, T);
    T sum_weigths_in_timestep(int);
    int rand_val_0_1(double p);
    void init_rdm_spike_train(int, int*, double);
    void evolve_net();
    void neuron_config_hom_setup();
    void weifgth_config_non_self_all_all();
    void set_initial_conditions();
    void print_config();
    void init_simulator();
    void rescale_constants_for_type();
};


// read in config from weigth_config.data and neuron_config.data
// this seems to destroy the saved state, the neuron needs to be initalized afterwards
template <class T>
void Neural_net_c<T>::read_in_config()
{
    std::ifstream fr ("weigth_config.data");
    std::string line;
    std::string word;
    int i = 0;
    int j = 0;
    double double_type;
    while (getline(fr, line))
    {
        std::istringstream iss(line);
        while(iss >> word)
        {
            // if not double: round the value
            // to get the best integer representation of the weigths
            this->nn.weigths[i][j] = typeid(T).name() == typeid(double_type).name() ?  std::stod(word) : std::round(std::stod(word));
            j++;
        }
        i++;
        j = 0;
    }
    fr.close();

    std::ifstream fr_config ("neuron_config.data");

    double value;

    for (int j_column = 0; j_column<columns; j_column++)
    {
        fr_config >> value;
        this->nn.neurons[j_column].constants.e_l_hidden = value;
        fr_config >> value;
        this->nn.neurons[j_column].constants.i_syn_0_hidden = value;
        fr_config >> value;
        this->nn.neurons[j_column].constants.tau_l_hidden = value;
        fr_config >> value;
        this->nn.neurons[j_column].constants.tau_syn_hidden = value;
        fr_config >> value;
        this->nn.neurons[j_column].constants.reset_potential_hidden = value;
        fr_config >> value;
        this->nn.neurons[j_column].constants.threshold_hidden = value;
        fr_config >> value;
        this->nn.neurons[j_column].constants.refrac_period_hidden = value;
    }
    fr_config.close();
}


template <class T> 
T Neural_net_c<T>::external_current(int step)
{
    T return_value;
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

template <class T> 
T Neural_net_c<T>::synaptic_input_right_side(T current, T weigthsum, int n_i, int *spiketimes, int step)
{
    // the 1/tau_syn is problematic for int -> solution: multiply and rescale equations - leading to transformed equations
    std::cout << weigthsum << "\n";
    std::cout << -1 * current / this->nn.neurons[n_i].constants.tau_syn
        + this->nn.neurons[n_i].constants.i_syn_0 * spiketimes[step] 
        + this->nn.neurons[n_i].constants.i_syn_0 * weigthsum << "\n";
    return -1 * current / this->nn.neurons[n_i].constants.tau_syn
        + this->nn.neurons[n_i].constants.i_syn_0 * spiketimes[step] 
        + this->nn.neurons[n_i].constants.i_syn_0 * weigthsum;
}

template <class T>
T Neural_net_c<T>::lif_right_side(T voltage, int n_i, int step)
{
    // the 1/tau_l is problematic for int
    return (this->nn.neurons[n_i].constants.e_l - voltage) / this->nn.neurons[n_i].constants.tau_l
        + external_current(step) 
        + this->nn.neurons[n_i].syn_current;
}

/*
template <class T>
neuron_state_var<T> Neural_net_c<T>::explicit_euler_step(T weigthsum, int n_i, int *spiketimes, int step, T d_t)
{
    T k_1_s = synaptic_input_right_side(this->nn.neurons[n_i].syn_current, weigthsum, n_i, spiketimes, step);
    this->nn.neurons[n_i].syn_current = this->nn.neurons[n_i].syn_current + k_1_s * d_t;

    T k_1_n = lif_right_side(this->nn.neurons[n_i].voltage, n_i, step);
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
*/


template <class T>
neuron_state_var<T> Neural_net_c<T>::runge_kutta_step(T weigthsum, int n_i, int *spiketimes, int step, T d_t)
{
    T k_1_s = synaptic_input_right_side(this->nn.neurons[n_i].syn_current, weigthsum, n_i, spiketimes, step);
    T k_2_s = synaptic_input_right_side(this->nn.neurons[n_i].syn_current + k_1_s * d_t, weigthsum, n_i, spiketimes, step);

    this->nn.neurons[n_i].syn_current = this->nn.neurons[n_i].syn_current + (k_1_s + k_2_s) / 2.0 * d_t;

    T k_1_n = lif_right_side(this->nn.neurons[n_i].voltage, n_i, step);
    T k_2_n = lif_right_side(this->nn.neurons[n_i].voltage + k_1_n * d_t, n_i, step);

    this->nn.neurons[n_i].voltage = this->nn.neurons[n_i].voltage + (k_1_n + k_2_n) / 2.0 * d_t;

    if (this->nn.neurons[n_i].refrac)
    {
        this->nn.neurons[n_i].voltage = this->nn.neurons[n_i].constants.reset_potential;
        this->nn.neurons[n_i].refrac_count += 1;
        if (this->nn.neurons[n_i].refrac_count > this->nn.neurons[n_i].constants.refrac_period / d_t)
            this->nn.neurons[n_i].refrac = 0;
        //printf("neuron is refrac \n");
        //printf("refrac_count: %d \n", this->nn.neurons[n_i].refrac_count);
    }

    if (this->nn.neurons[n_i].voltage > this->nn.neurons[n_i].constants.threshold)
    {
        this->nn.neurons[n_i].voltage = this->nn.neurons[n_i].constants.reset_potential;
        this->nn.neurons[n_i].fired = 1;
        this->nn.neurons[n_i].refrac = 1;
        printf("set refrac \n");
        this->nn.neurons[n_i].refrac_count = 0;
    }
    else
        this->nn.neurons[n_i].fired = 0;
    return this->nn.neurons[n_i];
}

template <class T>
T Neural_net_c<T>::sum_weigths_in_timestep(int neuron)
{
    T weight_sum = 0;
    for(int i=0; i<rows; i++)
    {
        if(this->nn.neurons[neuron].fired)
        {
            weight_sum += this->nn.weigths[neuron][i];
        }
    }
    return weight_sum;
}

template <class T>
int Neural_net_c<T>::rand_val_0_1(double p)
{
    double rand_double = (double)rand() / RAND_MAX;
    return rand_double > (1 - p);
}

template <class T>
void Neural_net_c<T>::init_rdm_spike_train(int num_sim_steps, int *spiketimes, double p)
{
    for(int i=0; i<num_sim_steps; i++)
    {
        spiketimes[i] = rand_val_0_1(p);
        if(spiketimes[i] == 1)
            printf("spike \n");
    }
}

template <class T>
void Neural_net_c<T>::evolve_net()
{
    std::ofstream fp ("simulator_output.sim_data");

    std::cout << this->num_sim_steps << "\n";

    T weigthsum;
    
    printf("refrac_count: %d \n", this->nn.neurons[0].refrac);

    for (int i=0; i<this->num_sim_steps; i++)
    {
        for (int j=0; j < columns; j++){
            weigthsum = sum_weigths_in_timestep(j);
            this->nn.neurons[j] = runge_kutta_step(weigthsum, j, this->spiketimes, this->step, this->d_t);
            fp << this->nn.neurons[j].voltage << " ";
            fp << this->nn.neurons[j].syn_current << " ";
            fp << this->nn.neurons[j].fired << " ";
        }
        this->t += this->d_t;
        this->step += 1;
        fp << this->t << "\n";
    }
    fp.close();
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

template <class T>
void Neural_net_c<T>::set_initial_conditions()
{
    for (int i=0; i<columns; i++)
    {
        this->nn.neurons[i].voltage = 0.0;
        this->nn.neurons[i].syn_current = 0.0;
        this->nn.neurons[i].refrac = 0;
        this->nn.neurons[i].refrac_count = 0;
        printf("initial condition set \n");
        printf("refrac_count: %d \n", this->nn.neurons[i].refrac_count);
    }
}

template <class T>
void Neural_net_c<T>::print_config()
{
    std::cout << "\nPrinting the weigth matrix \n";
    std::cout << "---------------------------- \n";
    for (int i=0; i<rows; i++)
    {
        for (int j=0; j<columns; j++)
        {
            std::cout << std::setprecision(6) << std::fixed << this->nn.weigths[i][j] << " ";
            if (j == columns - 1)
                std::cout << "\n";
        }
    }
    std::cout << "\nPrint config for neuron 0 \n";
    std::cout << "-----------------------------\n";
    std::cout << "e_l:            " << this->nn.neurons[0].constants.e_l << "\n";
    std::cout << "i_syn:          " << this->nn.neurons[0].constants.i_syn_0 << "\n";
    std::cout << "tau_l           " << this->nn.neurons[0].constants.tau_l << "\n";
    std::cout << "tau_syn         " << this->nn.neurons[0].constants.tau_syn << "\n";
    std::cout << "reset_potential " << this->nn.neurons[0].constants.reset_potential << "\n";
    std::cout << "threshold:      " << this->nn.neurons[0].constants.threshold << "\n";
    std::cout << "refrac_period:  " << this->nn.neurons[0].constants.refrac_period << "\n";
}

template <class T>
void Neural_net_c<T>::init_simulator()
{
    double double_type;
    if(typeid(T).name() == typeid(double_type).name() ? 1 : 0)
    {
        this->num_sim_steps = 200000;
        this->step = 0;
        this->d_t = 0.1e-3;
        this->t = 0.0;
    }
    else
    {
        this->num_sim_steps = 200000;
        this->step = 0;
        this->d_t = 1;
        this->t = 0;
    }
    init_rdm_spike_train(this->num_sim_steps, this->spiketimes, 0.0001);
}

template <class T>
void Neural_net_c<T>::rescale_constants_for_type()
{
    double double_type;
    if(typeid(T).name() == typeid(double_type).name() ? 1 : 0)
    {
        printf("double\n");
        for (int j_column = 0; j_column<columns; j_column++)
        {
            this->nn.neurons[j_column].constants.e_l = this->nn.neurons[j_column].constants.e_l_hidden;
            this->nn.neurons[j_column].constants.i_syn_0 = this->nn.neurons[j_column].constants.i_syn_0_hidden;
            this->nn.neurons[j_column].constants.tau_l = this->nn.neurons[j_column].constants.tau_l_hidden;
            this->nn.neurons[j_column].constants.one_over_tau_l = 1.0 / this->nn.neurons[j_column].constants.tau_l_hidden;
            this->nn.neurons[j_column].constants.tau_syn = this->nn.neurons[j_column].constants.tau_syn_hidden;
            this->nn.neurons[j_column].constants.one_over_tau_syn = 1.0 / this->nn.neurons[j_column].constants.tau_syn_hidden;
            this->nn.neurons[j_column].constants.reset_potential = this->nn.neurons[j_column].constants.reset_potential_hidden;
            this->nn.neurons[j_column].constants.threshold = this->nn.neurons[j_column].constants.threshold_hidden;
            this->nn.neurons[j_column].constants.refrac_period = this->nn.neurons[j_column].constants.refrac_period_hidden;
        }
    }
    else
    {
        printf("not double\n");
        // enter the translation to int here
        for (int j_column = 0; j_column<columns; j_column++)
        {
            double v_scaler = 1e6;
            double curr_scaler = 1e3;
            this->nn.neurons[j_column].constants.e_l = std::round(v_scaler * this->nn.neurons[j_column].constants.e_l_hidden);
            this->nn.neurons[j_column].constants.i_syn_0 = std::round(1e3 * this->nn.neurons[j_column].constants.i_syn_0_hidden);
            this->nn.neurons[j_column].constants.tau_l = std::round(curr_scaler * this->nn.neurons[j_column].constants.tau_l_hidden);
            this->nn.neurons[j_column].constants.one_over_tau_l = std::round(curr_scaler / this->nn.neurons[j_column].constants.tau_l_hidden);
            this->nn.neurons[j_column].constants.tau_syn = std::round(curr_scaler * this->nn.neurons[j_column].constants.tau_syn_hidden);
            this->nn.neurons[j_column].constants.one_over_tau_syn = std::round(curr_scaler / this->nn.neurons[j_column].constants.tau_syn_hidden);
            this->nn.neurons[j_column].constants.reset_potential = std::round(v_scaler * this->nn.neurons[j_column].constants.reset_potential_hidden);
            this->nn.neurons[j_column].constants.threshold = std::round(v_scaler * this->nn.neurons[j_column].constants.threshold_hidden);
            this->nn.neurons[j_column].constants.refrac_period = std::round(curr_scaler * this->nn.neurons[j_column].constants.refrac_period_hidden);
        }
    }
}

int main()
{
    Neural_net_c<int> nn;
    nn.init_simulator();
    nn.set_initial_conditions();
    nn.read_in_config();
    nn.set_initial_conditions();
    nn.rescale_constants_for_type();

    nn.print_config();

    nn.evolve_net();
}
