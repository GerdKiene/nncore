
module neuron
#(
    parameter WEIGTH_WIDTH = 6,
    parameter COUNTER_WIDTH = 16,
    parameter MEMBRANE_WIDTH = 16
)
(
    input logic clk,
    input logic [WEIGTH_WIDTH-1:0] syn_current,
    output logic spike_out,
    config_if.slave cfg_in
);

logic [MEMBRANE_WIDTH-1:0] membrane_voltage;

// define config registers -> unroll interface?


always_ff @(posedge clk) begin
    // open: how to implement the derivision
    membrane_voltage = (E_l - membrane_voltage) / tau_l + syn_current;
end

// as <= is executed concurrently, there might be race conditions lying here!
always_ff @(posedge clk) begin
    if(membrane_voltage > threshold) begin
        membrane_volate = reset_value;
        neuron_refrac <= 1'b1;
    end

    if(reset)
        refrac_counter <= 1'b0;
    if(neuron_refrac) begin
        refrac_counter <= refrac_counter + 1;
        if(refrac_counter > refrac_time)
            neuron_refrac <= 1'b0;
    end
end

endmodule
