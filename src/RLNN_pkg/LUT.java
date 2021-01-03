package RLNN_pkg;
import java.io.*;
import robocode.*;

public class LUT {
    private double[][] table;

    public int [][]visit;
    public LUT()
    {

        table=new double[Robo_States.Num][RoboActions.Num];
        visit=new int[Robo_States.Num][RoboActions.Num];
        initialize();
    }
    public void initialize()
    {
        for (int i=0;i<Robo_States.Num;i++)
            for(int j=0;j<RoboActions.Num; j++)
            {
                table[i][j]=0.0d;
                visit[i][j]=0;

            }
    }
    //for getting a Q-value for given state-action pair
    public double getValue(int state,int action)
    {
        return table[state][action];

    }

    //setting the value in the LUT
    public void setValue(int state,int action,double value)
    {
        table[state][action]=value;
    }
    public int getVisitTimes(int state,int action)
    {
        return visit[state][action];
    }

    //function for retrieving table
    public double [] [] getTable(){
        return table;
    }

    //For getting the maximum Q-value for a given state
    public double getMaxValue(int state)
    {
        double maxvalue=-100000000;
        for(int i=0;i<table[state].length;i++)
        {
            if (table[state][i]>maxvalue)
            {
                maxvalue=table[state][i];
            }
        }
        return maxvalue;
    }

    //Selecting the best action for a given state
    public int getBestAction(int state)
    {
        int action=0;
        double maxvalue=-1000000000;
        for(int i=0;i<table[state].length;i++)
        //for(int i=0;i<NumActions;i++)
        {
            if(table[state][i]>maxvalue)
            {
                maxvalue=table[state][i];
                action=i;
            }
        }
        return action;
    }

    //Loading LUT data
    public void loadData(File file)
    {
        BufferedReader r = null;
        try
        {
            r = new BufferedReader(new FileReader(file));
            for (int i = 0; i < Robo_States.Num; i++)
                for (int j = 0; j < RoboActions.Num; j++)
                    table[i][j] = Double.parseDouble(r.readLine());
        }
        catch (IOException e)
        {
            System.out.println("IOException trying to open reader: " + e);
            initialize();
        }
        catch (NumberFormatException e)
        {
            initialize();
        }
        catch(NullPointerException e){}
        finally
        {
            try
            {
                if (r != null)
                    r.close();
            }
            catch (IOException e)
            {
                System.out.println("IOException trying to close reader: " + e);
            }
        }
    }

    //Saving the LUT data
    public void saveData(File file)
    {
        PrintStream w = null;
        try
        {
            w = new PrintStream(new RobocodeFileOutputStream(file));
            for (int i = 0; i < Robo_States.Num; i++)
                for (int j = 0; j < RoboActions.Num; j++)
                    w.println(new Double(table[i][j]));

            if (w.checkError())
                System.out.println("Could not save the data!");
            w.close();
        }
        catch (IOException e)
        {
            System.out.println("IOException trying to write: " + e);
        }
        finally
        {
            try
            {
                if (w != null)
                    w.close();
            }
            catch (Exception e)
            {
                System.out.println("Exception trying to close writer: " + e);
            }
        }
    }
}
