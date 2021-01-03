package RLNN_pkg;

public class QLearning {
    public double learning_rate=0.2;
    public double discountRate=0.9;
    int preState;
    int preAction;
    double preValue;
    double newValue;

    LUT Qtable;
    boolean first=true;

    //initializing LUT
    public QLearning(LUT table)
    {
        this.Qtable=table;
    }

    //Function to perform the Reinforcement Learning as per LUT
    public void Learn(int state,int action,double reward,boolean policy)
    {
        if(first)
            first=false;
        else
        {
            //previous Q-value in the LUT
            preValue=Qtable.getValue(preState,preAction);

            //Off policy(Q-learning)
            if(policy==true)
            {
                newValue=(1-learning_rate)*preValue+learning_rate*(reward+discountRate*Qtable.getMaxValue(state));

            }

            //On policy(SARSA)
            else
            {
                newValue=(1-learning_rate)*preValue+learning_rate*(reward+discountRate*Qtable.getValue(state, action));
                Qtable.visit[preState][preAction]++;
            }

            Qtable.setValue(preState,preAction,newValue);
        }
        //storing state and action for future learning
        preState=state;
        preAction=action;

    }

    //selecting the action according to the exploration value
    public int selectAction(int state,double explore)
    {
        int action;
        double random=Math.random();
        if(random<explore)
        {
            //random action
            action=(int)(Math.random()*RoboActions.Num);
        }
        else
            //greedy move
            action=Qtable.getBestAction(state);
        return action;
    }
}
