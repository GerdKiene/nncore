#include <iostream>
#include <cmath>
#include <stdio.h>
#include <random>

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
