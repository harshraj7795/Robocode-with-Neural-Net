package RLNN_pkg;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;

public class NN_for_LUT {
    private static int numStateCategory = 5;    //state categories
    private static int numInput = numStateCategory;

    //NN hyper-parameters
    private static int numHidden = 40;
    private static int numOutput = 1;
    private static double expectedOutput[][]; //numStates*numActions
    private static double learningRate = 0.005;
    private static double momentumRate = 0.9;
    private static double lowerBound = -1.0;
    private static double upperBound = 1.0;
    private static double []maxQ = new double[RoboActions.Num];
    private static double []minQ = new double[RoboActions.Num];

    private static ArrayList<Double> errorInEachEpoch;
    private static ArrayList<NeuralNet> neuralNetworks;


    neuron_unit test_1 = new neuron_unit("test");

    public static void main(String[] args){
        LUT lut = new LUT();
        File file = new File("LUT.dat");
        lut.loadData(file);
        double inputData[][] = new double [Robo_States.Num][numStateCategory];
        double normExpectedOutput[][][] = new double [RoboActions.Num][Robo_States.Num][numOutput];
        expectedOutput = lut.getTable();

        for(int act = 0; act<RoboActions.Num;act++) {
            maxQ[act] = findMax(getColumn(expectedOutput,act));
            minQ[act] = findMin(getColumn(expectedOutput,act));
        }

        for(int stateid = 0; stateid < Robo_States.Num; stateid++) {
            int[]state = Robo_States.getStateFromIndex(stateid);
            inputData[stateid] = normalizeInputData(state);
            for(int act = 0; act < RoboActions.Num; act++) {
                normExpectedOutput[act][stateid][numOutput-1] =normalizeExpectedOutput(expectedOutput[stateid][act],maxQ[act],minQ[act],upperBound,lowerBound);
            }
        }
        neuralNetworks = new ArrayList<NeuralNet>();

        for(int act = 0; act < RoboActions.Num; act++) {
            int average = EpochAverage(act,inputData,normExpectedOutput[act],0.000005,10000,1);
            System.out.println(act+"Average of number of epochs to converge is: "+average+"\n");
        }

        for(NeuralNet net : neuralNetworks) {
            try {
                File weight = new File("Weight_"+net.getNetID()+".dat");
                weight.createNewFile();
                net.save(weight);
            }catch(IOException e) {
                System.out.println(e);
            }
        }

        System.out.println("End of test");

    }

    //normalizing the input data
    public static double [] normalizeInputData(int [] states) {
        double [] normalizedStates = new double [5];
        for(int i = 0; i < 5; i++) {
            switch (i) {
                case 0: //distance
                    normalizedStates[0] = -1.0 + ((double)states[0])*2.0/((double)(Robo_States.NumDistance-1));
                    break;
                case 1: //bearing
                    normalizedStates[1] = -1.0 + ((double)states[1])*2.0/((double)(Robo_States.NumBearing-1));;
                    break;
                case 2: //heading
                    normalizedStates[2] = -1.0 + ((double)states[2])*2.0/((double)(Robo_States.NumHeading-1));;
                    break;
                case 3: //hit_wall
                    normalizedStates[3] = -1.0 + ((double)states[3])*2.0;
                    break;
                case 4: //hit_bullet
                    normalizedStates[4] = -1.0 + ((double)states[4])*2.0;
                    break;
                default:
                    System.out.println("The data doesn't belong here.");
            }
        }
        return normalizedStates;
    }

    //normalizing the expected output
    public static double normalizeExpectedOutput(double expected, double max, double min, double upperbound, double lowerbound){
        double normalizedExpected = 0.0;
        if(expected > max) {
            expected = max;
        }else if(expected < min) {
            expected = min;
        }

        normalizedExpected = lowerbound +(expected-min)*(upperbound-lowerbound)/(max - min);


        return normalizedExpected;
    }

    public static double  remappingOutputToQ (double output, double max, double min, double upperbound, double lowerbound) {
        double remappedQ = 0.0;
        remappedQ = min + (output-lowerbound)*(max-min)/(upperbound - lowerbound);
        return remappedQ;
    }

    //function for calculating the average of no. of epochs for one training trial

    public static int EpochAverage(int act,double[][] input, double[][] expected,double minError, int maxSteps, int numTrials) {
        int epochNumber, failure,success;
        double average = 0f;
        epochNumber = 0;
        failure = 0;
        success = 0;
        NeuralNet testNeuronNet = null;
        for(int i = 0; i < numTrials; i++) {
            testNeuronNet = new NeuralNet(numInput,numHidden,numOutput,learningRate,momentumRate,lowerBound,upperBound,act ); //Construct a new neural net object
            tryConverge(testNeuronNet,input,expected,maxSteps, minError);
            epochNumber = getErrorArray().size();
            try {
                printRunResults(getErrorArray(), "Error_for_act_"+act+"_trail_"+numTrials+".csv");
            } catch (IOException e) {
                e.printStackTrace();
            }
            if( epochNumber < maxSteps) {
                average = average +  epochNumber;
                success ++;
            }
            else {
                failure++;
            }
        }
        double convergeRate = 100*success/(success+failure);
        System.out.println("The net converges for "+convergeRate+" percent of the time.\n" );
        average = average/success;
        neuralNetworks.add(testNeuronNet);
        return (int)average;
    }

    public static void tryConverge(NeuralNet theNet, double[][] input, double [][] expected,int maxStep, double minError) {
        int i;
        double totalerror = 1;
        double previousError = 0;
        errorInEachEpoch = new ArrayList<>();
        for(i = 0; i < maxStep && Math.abs(totalerror-previousError) > minError; i++) {
            previousError = totalerror;
            totalerror = 0.0;
            for(int j = 0; j < input.length; j++) {
                totalerror += theNet.train(input[j],expected[j]);
            }
            totalerror = Math.sqrt(totalerror/input.length);
            errorInEachEpoch.add(totalerror);
        }
        System.out.println("Sum of squared error in last epoch = " + totalerror);
        System.out.println("Number of epoch: "+ i + "\n");
        if(i == maxStep) {
            System.out.println("Error in training, try again!");
        }

    }

    //function for printing results
    public static void printRunResults(ArrayList<Double> errors, String fileName) throws IOException {
        int epoch;
        PrintWriter printWriter = new PrintWriter(new FileWriter(fileName));
        printWriter.printf("Epoch Number, Total Squared Error, \n");
        for(epoch = 0; epoch < errors.size(); epoch++) {
            printWriter.printf("%d, %f, \n", epoch, errors.get(epoch));
        }
        printWriter.flush();
        printWriter.close();
    }

    public static ArrayList <Double> getErrorArray(){
        return errorInEachEpoch;
    }

    public static void setErrorArray(ArrayList<Double> errors) {
        errorInEachEpoch = errors;
    }

    public static double findMax (double []theValues) {
        double maxQValue = theValues[0];
        for (int i = 0; i < theValues.length; i++) {
            if(maxQValue < theValues[i]) {
                maxQValue = theValues[i];
            }
        }
        return maxQValue;
    }

    public static double findMin (double []theValues) {
        double minQValue = theValues[0];
        for (int i = 0; i < theValues.length; i++) {
            if(minQValue > theValues[i]) {
                minQValue = theValues[i];
            }
        }
        return minQValue;
    }

    public static double[] getColumn(double[][] array, int index) {
        double[] column = new double[Robo_States.Num];
        for(int i = 0; i< column.length; i++ ) {
            column[i] = array[i][index];
        }
        return column;
    }

    public static int getNumInput() {
        return numInput;
    }

    public static int getNumHidden() {
        return numHidden;
    }

    public static int getNumOutput() {
        return numOutput;
    }

    public static double getLearningRate() {
        return learningRate;
    }

    public static double getMomentumRate() {
        return momentumRate;
    }
}
