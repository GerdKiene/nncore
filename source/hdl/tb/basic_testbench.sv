
parameter NUM_SYNAPSE_ROWS = 1;
parameter NUM_COLS = 1;


module basic_testbench();

localparam time timestep = 1us;

logic main_clk;

initial begin
    // load a sequence of test spikes, run the simulation
    #2000;
    $finish();
end

always begin
    main_clk = 1'b0;
    #(timestep / 2.0);
    main_clk = 1'b1;
    #(timestep / 2.0);
end

// instantiate the neurons and their connections

/* 
- read in config from txt file
- programm config registers
- initialize via reset
- run benchmark and put out same membrane traces
- implement priority encoder
- implement basic routing
*/

endmodule
