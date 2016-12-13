import numpy as np

weigth_config_file = open("weigth_config.data", "w")

rows = 10
columns = 10

for i in range(rows):
    for j in range(columns):
        if (i != j):
            weigth_config_file.write("1.1 ")
        if (i == j):
            weigth_config_file.write("0     ")
        if (j == columns - 1):
            weigth_config_file.write("\n")


neuron_config_file = open("neuron_config.data", "w")

constants = {"e_l": 0.0, "i_syn_0": 400.0, "tau_l": 0.01, "tau_syn": 3.0,\
        "reset_potential": 0.0, "threshold":1.2, "refrac_period": 1}

for i in range(columns):
    neuron_config_file.write("{0} {1} {2} {3} {4} {5} {6} \n".format(constants["e_l"], constants["i_syn_0"], constants["tau_l"], constants["tau_syn"], constants["reset_potential"], constants["threshold"], constants["refrac_period"]));


