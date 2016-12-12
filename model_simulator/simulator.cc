#include <iostream>
#include <cmath>
#include <stdio.h>
#include <iostream>
#include <iomanip>
#include <random>
#include <fstream>
#include <sstream>
#include <string>

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

template <typename T>
struct neural_net
{
    T weigths[rows][columns];
    neuron_state_var<T> neurons[columns];
};


template <class T>
class Neural_net_c
{
    neural_net<T> nn;
    T t;
    T d_t;
    int num_sim_steps;
    int step;
    int spiketimes[200000];


    public:
    void read_in_config();
    T external_current(int);
    T synaptic_input_right_side(T, T, int, int*, int);
    T lif_right_side(T, int, int);
    neuron_state_var<T> explicit_euler_step(T, int, int*, int, T);
    neuron_state_var<T> runge_kutta_step(T, int, int*, int, T);
    T sum_weigths_in_timestep(int);
    int rand_val_0_1(T p);
    void init_rdm_spike_train(int, int*, T);
    void evolve_net();
    void neuron_config_hom_setup();
    void weifgth_config_non_self_all_all();
    void set_initial_conditions();
    void print_config();
    void init_simulator();
};


template <class T>
void Neural_net_c<T>::read_in_config()
{
    std::ifstream fr ("weigth_config.data");
    std::string line;
    std::string word;
    int i = 0;
    int j = 0;
    while (getline(fr, line))
    {
        std::istringstream iss(line);
        i++;
        while(iss >> word)
        {
            j++;
            this->nn.weigths[i][j] = (T)std::round(std::stod(word));
        }
    }
    fr.close();

    std::ifstream fr_config ("neuron_config.data");

    double value;

    for (int j_column = 0; j_column<columns; j_column++)
    {
        fr_config >> value;
        this->nn.neurons[j_column].constants.e_l = (T)std::round(value);
        fr_config >> value;
        this->nn.neurons[j_column].constants.i_syn_0 = (T)std::round(value);
        fr_config >> value;
        this->nn.neurons[j_column].constants.tau_l = (T)std::round(value);
        fr_config >> value;
        this->nn.neurons[j_column].constants.tau_syn = (T)std::round(value);
        fr_config >> value;
        this->nn.neurons[j_column].constants.reset_potential = (T)std::round(value);
        fr_config >> value;
        this->nn.neurons[j_column].constants.threshold = (T)std::round(value);
        fr_config >> value;
        this->nn.neurons[j_column].constants.refrac_period = (T)std::round(value);
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
    return -1.0/this->nn.neurons[n_i].constants.tau_syn * current+ this->nn.neurons[n_i].constants.i_syn_0 * spiketimes[step] + this->nn.neurons[n_i].constants.i_syn_0 * weigthsum;
}

template <class T>
T Neural_net_c<T>::lif_right_side(T voltage, int n_i, int step)
{
    return 1.0/this->nn.neurons[n_i].constants.tau_l * (this->nn.neurons[n_i].constants.e_l - voltage) + external_current(step) + this->nn.neurons[n_i].syn_current;
}

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
int Neural_net_c<T>::rand_val_0_1(T p)
{
    double rand_double = (double)rand() / RAND_MAX;
    return rand_double > (1 - p);
}

template <class T>
void Neural_net_c<T>::init_rdm_spike_train(int num_sim_steps, int *spiketimes, T p)
{
    for(int i=0; i<num_sim_steps; i++)
    {
        spiketimes[i] = rand_val_0_1(p);
    }
}

template <class T>
void Neural_net_c<T>::evolve_net()
{
    std::ofstream fp ("simulator_output.sim_data");

    std::cout << this->num_sim_steps << "\n";

    T weigthsum;
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
    this->num_sim_steps = 200000;
    this->step = 0;
    this->d_t = 0.1e-3;
    this->t = 0.0;
    init_rdm_spike_train(this->num_sim_steps, this->spiketimes, 1500.0 * this->d_t);
}

int main()
{
    Neural_net_c<int> nn;
    nn.init_simulator();
    nn.set_initial_conditions();
    nn.read_in_config();
    
    nn.print_config();

    nn.evolve_net();
}
