package RLNN_pkg;
import java.util.ArrayList;
import java.util.Random;
import java.io.*;
import robocode.RobocodeFileOutputStream;

public class NeuralNet {
    private int netID;
    private int argNumInputs;
    private int argNumHiddens;
    private int argNumOutputs;
    private double argLearningRate;
    private double argMomentumRate;
    private double argQMin;
    private double argQMax;

    //Defining array lists for inputs, hidden neurons and outputs
    private ArrayList<neuron_unit> inputLayerNeurons = new ArrayList<neuron_unit>();
    private ArrayList<neuron_unit> hiddenLayerNeurons = new ArrayList<neuron_unit>();
    private ArrayList<neuron_unit> outputLayerNeurons = new ArrayList<neuron_unit>();


    private neuron_unit biasNeuron = new neuron_unit("bias"); // For bias

    private double epochOutput[][] = {{-1},{-1}, {-1}, {-1}};//Initial value is -1 for each output
    private ArrayList<Double> errorInEachEpoch = new ArrayList<>();
    private ArrayList<Integer> epochEachTrail = new ArrayList<>();

    public NeuralNet(
            int numInputs, int numHiddens,
            int numOutputs, double learningRate,
            double momentumRate, double a, double b,
            int id
    ) {
        this.argNumInputs = numInputs;
        this.argNumHiddens = numHiddens;
        this.argNumOutputs = numOutputs;
        this.argLearningRate = learningRate;
        this.argMomentumRate = momentumRate;
        this.argQMin = a;
        this.argQMax = b;

        this.setUpNetwork();
        this.initializeWeights();
        this.netID = id;
    }

    public void setUpNetwork() {
        //Setting up the Inputs
        for(int i = 0; i < this.argNumInputs;i++) {
            String index = "Input"+Integer.toString(i);
            neuron_unit neuron = new neuron_unit(index);
            inputLayerNeurons.add(neuron);
        }
        biasNeuron.setOutput(1.0);

        for(int j = 0; j < this.argNumHiddens;j++) {
            String index = "Hidden"+Integer.toString(j);
            neuron_unit neuron = new neuron_unit(index,"Customized",inputLayerNeurons,biasNeuron);
            hiddenLayerNeurons.add(neuron);
        }

        for(int k = 0; k < this.argNumOutputs;k++) {
            String index = "Output"+Integer.toString(k);
            neuron_unit neuron = new neuron_unit(index,"Customized",hiddenLayerNeurons,biasNeuron);
            outputLayerNeurons.add(neuron);
        }
    }

    //setting the input value in feedforward
    public void setInputData(double [] inputs) {
        for(int i = 0; i < inputLayerNeurons.size(); i++) {
            inputLayerNeurons.get(i).setOutput(inputs[i]);//Input Layer Neurons only have output values.
        }
    }

    //Getting Output Results
    public double[] getOutputResults() {
        double [] outputs = new double[outputLayerNeurons.size()];
        for(int i = 0; i < outputLayerNeurons.size(); i++) {
            outputs[i] = outputLayerNeurons.get(i).getOutput();
        }
        return outputs;
    }

    public int getNetID(){
        return this.netID;
    }


    //Performing forward propagation step
    public void forwardPropagation() {
        for(neuron_unit hidden: hiddenLayerNeurons) {
            hidden.calculateOutput(argQMin,argQMax);
        }

        for (neuron_unit output: outputLayerNeurons) {
            output.calculateOutput(argQMin, argQMax);
        }
    }

    public ArrayList<neuron_unit> getInputNeurons(){
        return this.inputLayerNeurons;
    }

    public ArrayList<neuron_unit> getHiddenNeurons(){
        return this.hiddenLayerNeurons;
    }

    public ArrayList<neuron_unit> getOutputNeurons(){
        return this.outputLayerNeurons;
    }

    public ArrayList<Double> getErrorArray(){
        return this.errorInEachEpoch;
    }

    public ArrayList<Integer> getEpocNumArray(){
        return this.epochEachTrail;
    }


    public double [][] getEpochResults() {
        return epochOutput;
    }

    public void setEpochResults(double[][] results){
        for(int i = 0; i < results.length;i++) {
            for(int j = 0; j < results[i].length;j++)
            {
                epochOutput[i][j] = results[i][j];
            }
        }
    }
    //backpropagation
    private void applyBackpropagation(double expectedOutput[]) {
        int i = 0;
        for(neuron_unit output : outputLayerNeurons) {
            double yi = output.getOutput();
            double ci = expectedOutput[i];

            ArrayList<NN_connection> connections = output.getInputConnectionList();
            for(NN_connection link : connections) {
                double xi = link.getInput();
                double error = customSigmoidDerivative(yi)*(ci-yi);
                link.setError(error);
                double deltaWeight = argLearningRate*error*xi + argMomentumRate*link.getDeltaWeight();
                double newWeight = link.getWeight() + deltaWeight;
                link.setDeltaWeight(deltaWeight);
                link.setWeight(newWeight);
            }
            i++;
        }

        for(neuron_unit hidden: hiddenLayerNeurons) {
            ArrayList<NN_connection> connections = hidden.getInputConnectionList();
            double yi =hidden.getOutput();
            for(NN_connection link : connections) {
                double xi = link.getInput();
                double sumWeightedError= 0;
                for(neuron_unit output: outputLayerNeurons) {
                    double wjh = output.getInputConnection(hidden.getId()).getWeight();
                    double errorFromAbove = output.getInputConnection(hidden.getId()).getError();
                    sumWeightedError = sumWeightedError + wjh *errorFromAbove;
                }

                double error = customSigmoidDerivative(yi)*sumWeightedError;
                link.setError(error);
                double deltaWeight = argLearningRate*error*xi + argMomentumRate * link.getDeltaWeight();
                double newWeight = link.getWeight() + deltaWeight;
                link.setDeltaWeight(deltaWeight);
                link.setWeight(newWeight);
            }
        }
    }


    public double [] outputFor(double[] inputData) {
        setInputData(inputData);
        forwardPropagation();
        double outputs[] = getOutputResults();
        return outputs;
    }



    //Function to perform the training of NN
    public double train(double [] argInputVector, double [] argTargetOutput) {
        double error = 0.0;
        double output[] = outputFor(argInputVector);
        for (int j = 0; j < argTargetOutput.length; j++) {
            double deltaErr = Math.pow((output[j]-argTargetOutput[j]),2);
            error = error + deltaErr;//sum of error for all  output neurons
        }
        this.applyBackpropagation(argTargetOutput);
        return error;
    }


    //Functions for printing, saving and loading results
    public void printEachTrail(ArrayList<Integer> epochNum, String fileName) throws IOException {
        PrintWriter printWriter = new PrintWriter(new FileWriter(fileName));
        printWriter.printf("Trail Number, Total epoch each Trail, \n");

        for(int trail = 0; trail < epochNum.size(); trail++) {
            printWriter.printf("%d, %d, \n", trail, epochNum.get(trail));
        }
        printWriter.flush();
        printWriter.close();
    }


    public void save(File argFile) {
        PrintStream savefile = null;
        try{
            savefile = new PrintStream(new FileOutputStream(argFile,false) );
            savefile.println(outputLayerNeurons.size());
            savefile.println(hiddenLayerNeurons.size());
            savefile.println(inputLayerNeurons.size());
            for(neuron_unit output : outputLayerNeurons){
                ArrayList<NN_connection> connections = output.getInputConnectionList();
                for(NN_connection link : connections){
                    savefile.println(link.getWeight());
                }
            }
            for(neuron_unit hidden: hiddenLayerNeurons) {
                ArrayList<NN_connection> connections = hidden.getInputConnectionList();
                for(NN_connection link : connections){
                    savefile.println(link.getWeight());
                }
            }
            savefile.flush();
            savefile.close();
        }
        catch(IOException e){
            System.out.println("Cannot save the weight table.");
        }

    }

    public void save_robot(File argFile) {
        PrintStream savefile = null;
        try{
            savefile = new PrintStream(new RobocodeFileOutputStream(argFile));
            savefile.println(outputLayerNeurons.size());
            savefile.println(hiddenLayerNeurons.size());
            savefile.println(inputLayerNeurons.size());
            for(neuron_unit output : outputLayerNeurons){
                ArrayList<NN_connection> connections = output.getInputConnectionList();
                for(NN_connection link : connections){
                    savefile.println(link.getWeight());
                }
            }
            for(neuron_unit hidden: hiddenLayerNeurons) {
                ArrayList<NN_connection> connections = hidden.getInputConnectionList();
                for(NN_connection link : connections){
                    savefile.println(link.getWeight());
                }
            }
            savefile.flush();
            savefile.close();
        }
        catch(IOException e){
            System.out.println("Cannot save the weight table.");
        }

    }


    public void load(File argFileName) throws IOException {

        try{
            BufferedReader readfile = new BufferedReader(new FileReader(argFileName));
            int numOutputNeuron = Integer.valueOf(readfile.readLine());
            int numHiddenNeuron = Integer.valueOf(readfile.readLine());
            int numInputNeuron = Integer.valueOf(readfile.readLine());
            if ( numInputNeuron != inputLayerNeurons.size() ) {
                System.out.println ( "*** Number of inputs in file does not match expectation");
                readfile.close();
                throw new IOException();
            }
            if ( numHiddenNeuron != hiddenLayerNeurons.size() ) {
                System.out.println ( "*** Number of hidden in file does not match expectation" );
                readfile.close();
                throw new IOException();
            }
            if ( numOutputNeuron != outputLayerNeurons.size() ) {
                System.out.println ( "*** Number of output in file does not match expectation" );
                readfile.close();
                throw new IOException();
            }

            for(neuron_unit output : outputLayerNeurons){
                ArrayList<NN_connection> connections = output.getInputConnectionList();
                for(NN_connection link : connections){
                    link.setWeight(Double.valueOf(readfile.readLine()));
                }
            }
            for(neuron_unit hidden: hiddenLayerNeurons) {
                ArrayList<NN_connection> connections = hidden.getInputConnectionList();
                for(NN_connection link : connections){
                    link.setWeight(Double.valueOf(readfile.readLine()));
                }
            }

            readfile.close();
        }
        catch(IOException e){
            System.out.println("IOException failed to open reader: " + e);
        }

    }


    //function for derivative of sigmoid
    public double sigmoidDerivative(double yi) {
        double result = yi*(1 - yi);
        return result;
    }

    //function for derivative of bipolar sigmoid
    public double bipolarSigmoidDerivative(double yi) {
        double result = 1.0/2.0 * (1-yi) * (1+yi);
        return result;
    }

    //function for derivative of custom sigmoid
    public double customSigmoidDerivative(double yi) {
        double result = -(1.0/(argQMax-argQMin)) * (yi-argQMin) * (yi-argQMax);
        return result;
    }

    public void initializeWeights() {

        double upperbound = 0.5;
        double lowerbound = -0.5;
        for(neuron_unit neuron: hiddenLayerNeurons) {
            ArrayList <NN_connection> connections = neuron.getInputConnectionList();
            for(NN_connection connect: connections) {
                connect.setWeight(getRandom(lowerbound,upperbound));
            }
        }
        for(neuron_unit neuron: outputLayerNeurons) {
            ArrayList <NN_connection> connections = neuron.getInputConnectionList();
            for(NN_connection connect: connections) {
                connect.setWeight(getRandom(lowerbound,upperbound));
            }
        }
    }


    public void zeroWeights() {
        for(neuron_unit neuron: hiddenLayerNeurons) {
            ArrayList <NN_connection> connections = neuron.getInputConnectionList();
            for(NN_connection connect: connections) {
                connect.setWeight(0);
            }
        }
        for(neuron_unit neuron:outputLayerNeurons) {
            ArrayList <NN_connection> connections = neuron.getInputConnectionList();
            for(NN_connection connect: connections) {
                connect.setWeight(0);
            }
        }

    }

    public double getRandom(double lowerbound, double upperbound) {
        double random = new Random().nextDouble();

        double result = lowerbound+(upperbound-lowerbound)*random;
        return result;
    }

    public double getBias() {
        return biasNeuron.getOutput();
    }
}
