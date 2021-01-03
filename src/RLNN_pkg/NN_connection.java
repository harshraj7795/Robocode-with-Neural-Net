package RLNN_pkg;

public class NN_connection {

    //source of the connection
    private neuron_unit SrcNeuron;
    // destination of the connection
    private neuron_unit DestNeuron;

    private double currentInput = 0;
    //weight of the connection
    private double weight = 0;

    private double deltaWeight = 0;

    private double prevDeltaWeight = 0;

    private double error = 0;


    //Construction with random initializations
    public NN_connection(neuron_unit src, neuron_unit dest) {
        this.SrcNeuron = src;
        this.DestNeuron = dest;
    }

    public NN_connection(neuron_unit src, neuron_unit dest, double weight) {
        this.SrcNeuron = src;
        this.DestNeuron = dest;
        this.weight = weight;
    }

    public double getWeight() {
        return this.weight;
    }


    public void setWeight(double weight) {
        this.weight = weight;
    }

    public double getDeltaWeight() {
        return this.deltaWeight;
    }

    public void setDeltaWeight(double delta) {
        this.prevDeltaWeight = this.deltaWeight;//Record previous delta weight
        this.deltaWeight = delta;
    }

    public double getPrevDeltaWeight() {
        return this.prevDeltaWeight;
    }

    public void setError(double value) {
        this.error = value;
    }

    public double getError() {
        return this.error;
    }


    public double getInput() {
        this.currentInput = SrcNeuron.getOutput();
        return this.currentInput;
    }


    public neuron_unit getSrcNeuron() {
        return this.SrcNeuron;
    }

    public neuron_unit getDestNeuron() {
        return this.DestNeuron;
    }
}
