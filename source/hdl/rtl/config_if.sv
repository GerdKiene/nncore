interface config_if();
    parameter DATA_WIDTH = 8;
    logic [DATA_WIDTH-1:0] data_in, data_out;

    modport master(output data_in, input data_out, output data_clk);
    modport slave(input data_in, output data_out, input data_clk);
endinterface
