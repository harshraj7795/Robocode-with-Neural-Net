package RLNN_pkg;
import java.util.ArrayList;
import java.util.List;
import java.util.HashMap;

public class neuron_unit {
    final String id;
    //for saving input connections
    private ArrayList <NN_connection> inputConnections = new ArrayList <NN_connection> ();
    // hashmap for saving the incoming connections for lookup
    private HashMap<String, NN_connection> inputconnectionMap = new HashMap<String, NN_connection>();

    private String activationType;

    final double bias = 1;
    public double NeuronOutput = 0;

    //Constructor
    public neuron_unit(String id) {
        this.id = id;
    }

    public neuron_unit(String id, String ActivationFunctionType, List<neuron_unit> inputNeurons) {
        this.id = id;
        setActivationType(ActivationFunctionType);
        addInputConnections(inputNeurons);
    }

    public neuron_unit(String id, String ActivationFunctionType, List<neuron_unit> inputNeurons, neuron_unit bias) {
        this.id = id;
        setActivationType(ActivationFunctionType);
        addInputConnections(inputNeurons);
        addBiasConnection(bias);
    }

    //Functions for utility purpose
    public String getId() {
        return this.id;
    }

    public void setActivationType(String type) {
        this.activationType = type;
    }

    public String getActivationType() {
        return this.activationType;
    }

    //functions for calculating outputs
    public double getOutput() {
        return this.NeuronOutput;
    }

    public void setOutput(double output) {
        this.NeuronOutput = output;
    }


    public void calculateOutput(double ArgA, double ArgB) {
        double weightedSum = inputSummingFunction(this.inputConnections);
        if(this.activationType == "bipolar") {
            this.NeuronOutput = bipolarSigmoid(weightedSum);
        }else if(this.activationType == "Unipolar") {
            this.NeuronOutput = unipolarSigmoid(weightedSum);
        }else if(this.activationType == "Customized") {
            this.NeuronOutput = customizedSigmoid(weightedSum,ArgA,ArgB);
        }
    }


    private double inputSummingFunction(ArrayList<NN_connection> inputConnections) {
        double weightedSum = 0;
        for(NN_connection connection : inputConnections)
        {
            double weight = connection.getWeight();
            double input = connection.getInput();
            weightedSum = weightedSum + weight*input;
        }

        return weightedSum;
    }

    //Functions for Connections
    private void addInputConnections(List<neuron_unit> inputNeurons) {
        for(neuron_unit neuron : inputNeurons) {
            NN_connection connection = new NN_connection(neuron,this);//creating a new connection that connects with the neuron
            inputConnections.add(connection);//Putting the connection into the array
            inputconnectionMap.put(neuron.getId(), connection);//Putting the created connection into the hash map
        }
    }

    private void addBiasConnection(neuron_unit neuron) {
        NN_connection connection = new NN_connection(neuron,this);
        inputConnections.add(connection); //Adding bias connection to the list
    }

    public NN_connection getInputConnection(String neuronId) {
        return inputconnectionMap.get(neuronId);
    }

    public ArrayList<NN_connection> getInputConnectionList() {
        return this.inputConnections;
    }



    public double unipolarSigmoid(double weightedSum){

        return 1/(1 + Math.exp(-weightedSum));
    }
    public double bipolarSigmoid(double weightedSum) {
        return 2/(1 + Math.exp(-weightedSum))-1;
    }

    public double customizedSigmoid(double weightedSum,double a, double b){
        return (b-a)/(1+Math.exp(-weightedSum))+a;
    }
}
