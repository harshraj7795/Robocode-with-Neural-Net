package RLNN_pkg;
import java.util.ArrayList;
import java.util.Random;

public class Learning {
    //parameters for RL
    public static final double discountRate = 0.9;
    public static double explorationRate = 0.2;     //probability of selecting random move
    private static double learningRate = 0.2; // alpha
    private int currentState;
    private int currentAction;
    private boolean firstRound = true;
    private LUT table;


    private static int numStateCategory = 5;
    private static int numInput = numStateCategory;

    //NN hyper-parameters
    private static int numHidden = NN_for_LUT.getNumHidden();
    private static int numOutput = NN_for_LUT.getNumOutput();
    private static double learningRate_NN = NN_for_LUT.getLearningRate();
    private static double momentumRate = NN_for_LUT.getMomentumRate();
    private static double lowerBound = -1.0;
    private static double upperBound = 1.0;

    private double []maxQ = new double[RoboActions.Num];
    private double []minQ = new double[RoboActions.Num];


    private int[] currentStateArray = new int [numStateCategory];
    private int[] newStateArray = new int [numStateCategory];
    private double [] currentActionOutput  = new double [RoboActions.Num];
    private double [] newActionOutput = new double[RoboActions.Num];
    private double [] currentActionQ  = new double[RoboActions.Num];
    private double [] newActionQ = new double[RoboActions.Num];
    private double qError[][]= new double [Robo_States.Num][RoboActions.Num];

    private ArrayList<NeuralNet> neuralNetworks = new ArrayList<NeuralNet>();
    public Learning(LUT table) {
        this.table = table;
        for (int act = 0; act<RoboActions.Num; act++) {
            maxQ[act] = 15;
            minQ[act] = -5;
        }
    }

    // Off-policy
    public void QLearn (int nextState, int nextAction, double reward) {
        double oldQ;
        double newQ;
        if (firstRound) {
            firstRound = false;
        } else {
            oldQ = table.getValue(currentState, currentAction);
            newQ = oldQ + learningRate*(reward + discountRate * table.getMaxValue(nextState)-oldQ);
            table.setValue(currentState, currentAction, newQ);

        }
        currentState = nextState;
        currentAction = nextAction;

    }

    // On-policy
    public void SARSALearn (int nextState, int nextAction, double reward) {
        double oldQ;
        double newQ;
        if (firstRound) {
            firstRound = false;
        } else {
            oldQ = table.getValue(currentState, currentAction);
            newQ = oldQ + learningRate*(reward + discountRate * table.getValue(nextState, nextAction)-oldQ);
            table.setValue(currentState, currentAction, newQ);

        }
        currentState = nextState;
        currentAction = nextAction;
    }

    // function for selecting the action
    public int selectAction(int state, boolean isOnline) {
        double thres = Math.random();
        int action = 0;
        double [] inputData = normalizeInputData(currentStateArray);
        if (thres < explorationRate) {
            // Exploratory move
            Random ran = new Random();
            action = ran.nextInt(RoboActions.Num);
        } else {
            // Greedy move
            if (isOnline) {
                // input sent into 6 NN and get 6 output
                for(NeuralNet theNet : neuralNetworks) {
                    int act = theNet.getNetID();
                    double currentNetOutput = theNet.outputFor(inputData)[0];
                    double currentNetQValue = remappingOutputToQ(currentNetOutput, maxQ[act], minQ[act], upperBound, lowerBound);//Reverse map output to big scale
                    int currentNetIndex = theNet.getNetID();
                    setCurrentActionValue(currentNetOutput,currentNetIndex);//Probably wrong
                    setCurrentQValue(currentNetQValue,currentNetIndex);
                }
                action = getMaxIndex(currentActionQ);
            } else {
                action = table.getBestAction(state);
            }
        }
        return action;
    }


    //Combining RL with NN

    public void nn_QLearn( int state, int action, double reward) {

        double currentStateQValue = getCurrentQValues()[action] ;
        double [] newInputData = new double[numStateCategory];
        newInputData = NN_for_LUT.normalizeInputData(getNewStateArray());
        for(NeuralNet theNet: neuralNetworks) {
            int act = theNet.getNetID();
            double tempOutput = theNet.outputFor(newInputData)[0];
            double tempQValue = remappingOutputToQ(tempOutput, maxQ[act], minQ[act], upperBound, lowerBound);
            setNewActionValue(tempOutput, theNet.getNetID());
            setNewQValue(tempQValue, theNet.getNetID());
        }

        int maxNewStateActionIndex = getMaxIndex(getNewActionValues());
        double maxNewQValue = getNewQValues()[maxNewStateActionIndex];
        double expectedQValue = currentStateQValue + learningRate*(reward + discountRate *maxNewQValue -currentStateQValue);
        double [] expectedOutput = new double[1];
        expectedOutput[0] = NN_for_LUT.normalizeExpectedOutput(expectedQValue, maxQ[action], minQ[action], upperBound, lowerBound);
        NeuralNet learningNet = neuralNetworks.get(action);
        double [] currentInputData = NN_for_LUT.normalizeInputData(getNewStateArray());
        learningNet.train(currentInputData, expectedOutput);
        if (getCurrentQValues()[action] != 0) {
            qError[state][action] = (getNewQValues()[action] - getCurrentQValues()[action])/getCurrentQValues()[action];
        }

    }

    public void setCurrentStateArray (int state) {
        currentStateArray = Robo_States.getStateFromIndex(state);
    }
    public void setNewStateArray (int state) {
        newStateArray = Robo_States.getStateFromIndex(state);
    }

    public void initializeNeuralNetworks(){
        for(int i = 0; i < RoboActions.Num; i++) {
            NeuralNet theNewNet = new NeuralNet(numInput,numHidden,numOutput,learningRate_NN,momentumRate,lowerBound,upperBound,i);
            neuralNetworks.add(theNewNet);
        }
    }

    //Function to normalize the input data
    public static double [] normalizeInputData(int [] states) {
        double [] normalizedStates = new double [5];
        for(int i = 0; i < 5; i++) {
            switch (i) {
                case 0:
                    normalizedStates[0] = -1.0 + ((double)states[0])*2.0/((double)(Robo_States.NumDistance-1));
                    break;
                case 1:
                    normalizedStates[1] = -1.0 + ((double)states[1])*2.0/((double)(Robo_States.NumBearing-1));;
                    break;
                case 2:
                    normalizedStates[2] = -1.0 + ((double)states[2])*2.0/((double)(Robo_States.NumHeading-1));;
                    break;
                case 3:
                    normalizedStates[3] = -1.0 + ((double)states[3])*2.0;
                    break;
                case 4:
                    normalizedStates[4] = -1.0 + ((double)states[4])*2.0;
                    break;
                default:
                    System.out.println("The data doesn't belong here.");
            }
        }
        return normalizedStates;
    }

    public static double normalizeExpectedOutput(double expected, double max, double min, double upperbound, double lowerbound){
        double normalizedExpected = 0.0;
        double localExpected = expected;
        if(localExpected > max) {
            localExpected = max;
        }else if(localExpected < min) {
            localExpected = min;
        }

        normalizedExpected = lowerbound +(localExpected-min)*(upperbound-lowerbound)/(max - min);


        return normalizedExpected;
    }

    public static double  remappingOutputToQ (double output, double maxQ, double minQ, double upperbound, double lowerbound) {
        double remappedQ = 0.0;
        double currentOutput = output;
        if(currentOutput < -1.0) {
            currentOutput = -1.0;
        }else if(currentOutput > 1.0) {
            currentOutput = 1.0;
        }
        remappedQ = minQ + (currentOutput-lowerbound)/(upperbound-lowerbound)*(maxQ - minQ);
        return remappedQ;
    }

    public static double[] getColumn(double[][] array, int index) {
        double[] column = new double[Robo_States.Num];
        for(int i = 0; i< column.length; i++ ) {
            column[i] = array[i][index];
        }
        return column;
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

    public int getMaxIndex(double [] theValues) {
        double maxQValue = theValues[0];
        int maxIndex = 0;
        for(int i = 0; i < theValues.length; i++) {
            if(maxQValue < theValues[i]) {
                maxQValue = theValues[i];
                maxIndex = i;
            }
        }
        return maxIndex;
    }

    public void setCurrentActionValue(double theValues, int theIndex) {
        currentActionOutput[theIndex] = theValues;
    }
    public double [] getCurrentActionValues() {
        return this.currentActionOutput;
    }

    public void setCurrentQValues(double [] theValues) {
        currentActionQ = theValues;
    }
    public void setCurrentQValue(double theValues, int theIndex) {
        currentActionQ[theIndex] = theValues;
    }

    public double [] getCurrentQValues() {
        return this.currentActionQ;
    }

    public void setNewQValues(double [] theValues) {
        newActionQ = theValues;
    }
    public void setNewQValue(double theValues, int theIndex) {
        newActionQ[theIndex] = theValues;
    }
    public double [] getNewQValues() {
        return this.newActionQ;
    }

    public int [] getNewStateArray(){
        return this.newStateArray;
    }

    public void setCurrentActionValues(double [] theValues) {
        currentActionOutput = theValues;
    }


    public void setNewActionValues(double [] theValues) {
        newActionOutput = theValues;
    }
    public void setNewActionValue(double theValues, int theIndex) {
        newActionOutput[theIndex] = theValues;
    }
    public double [] getNewActionValues() {
        return this.newActionOutput;
    }
    public ArrayList<NeuralNet> getNeuralNetworks(){
        return this.neuralNetworks;

    }

    public double getQError (int state, int action) {
        return qError[state][action];
    }

    public void setQError (int state, int action, double value) {
        qError[state][action] = value;
    }

}
