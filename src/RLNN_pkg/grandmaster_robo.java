package RLNN_pkg;

import robocode.AdvancedRobot;
import java.awt.*;
import java.awt.geom.*;
import java.io.*;
import robocode.*;
import java.util.ArrayList;

public class grandmaster_robo extends AdvancedRobot {
    public static final double PI = Math.PI;
    private Target target;
    private static LUT table;
    private Learning learner;

    //initializing robot battle parameters
    private double reward=0.0 ;
    private double accu_reward=0.0;
    private double firePower;
    private int isHitWall = 0;
    private int isHitByBullet = 0;
    private double error_val;
    ArrayList<NeuralNet> templist = new ArrayList<NeuralNet>();


    //============================================================
    //Starting the robot tank battle
    //============================================================
    public void run()
    {
        //Initializing the objects for implementing RL
        table = new LUT();
        learner = new Learning(table);

        target = new Target();
        target.distance = 100000;


        //Initial Setting of Robot Parameters
        setColors(Color.red, Color.blue, Color.gray);
        setAdjustGunForRobotTurn(true);
        setAdjustRadarForGunTurn(true);
        turnRadarRightRadians(2 * PI);

        //initialization of Neural Nets
        learner.initializeNeuralNetworks();
        templist=learner.getNeuralNetworks();

        //initialization of robot weights
        for(NeuralNet theNet: learner.getNeuralNetworks()) {
            try {
                theNet.load(getDataFile("Weight_"+theNet.getNetID()+".dat"));
            } catch (IOException e) {
                e.printStackTrace();
            }
        }

        //Into the Battle Loop
        while (true)
        {
            robotMovement();    //function for robot movement

            //setting firepower
            firePower = 400 / target.distance;
            if (firePower > 3)
            {
                firePower = 3;
            }
            radarMovement();    //function for radar movement
            gunMovement();      //function for gun movement
            if (getGunHeat() == 0)
            {
                setFire(firePower);     //setting firepower
            }
            execute();
        }
    }

    //====================================================================
    //Functions for Robot Movement, Radar Movement, Gun Movement

    private void robotMovement()
    {

        int state = getState();
        learner.setNewStateArray(state);
        //selecting the action
        int action=learner.selectAction(state,true);

        //online learning during robocode battle
        learner.nn_QLearn(state,action,reward);
        //calculating error for q-values
        if (Robo_States.getHeading(getHeading())==2 && action==0){
            error_val=learner.getQError(state,action);
        }
        accu_reward+=reward;
        reward = 0.0;
        isHitWall = 0;
        isHitByBullet = 0;

        //actions to be performed
        switch (action)
        {
            case RoboActions.RobotAhead:
                setAhead(RoboActions.RobotMoveDistance1);
                break;
            case RoboActions.RobotBack:
                setBack(RoboActions.RobotMoveDistance2);
                break;
            case RoboActions.RobotAheadTurnLeft:
                setAhead(RoboActions.RobotMoveDistance1);
                setTurnLeft(RoboActions.RobotTurnDegree);
                break;
            case RoboActions.RobotAheadTurnRight:
                setAhead(RoboActions.RobotMoveDistance1);
                setTurnRight(RoboActions.RobotTurnDegree);
                break;
            case RoboActions.RobotBackTurnLeft:
                setBack(RoboActions.RobotMoveDistance2);
                setTurnLeft(RoboActions.RobotTurnDegree);
                break;
            case RoboActions.RobotBackTurnRight:
                setBack(RoboActions.RobotMoveDistance2);
                setTurnRight(RoboActions.RobotTurnDegree);
                break;
        }
    }

    //radar movement function
    private void radarMovement()
    {
        setTurnRadarRightRadians(Double.POSITIVE_INFINITY);
    }

    //gun movement function
    private void gunMovement()
    {


        long time;
        long nextTime;
        Point2D.Double p;
        p = new Point2D.Double(target.x, target.y);
        for (int i = 0; i < 20; i++)
        {
            nextTime = (int)Math.round((getrange(getX(),getY(),p.x,p.y)/(20-(3*firePower))));
            time = getTime() + nextTime - 10;
            p = target.guessPosition(time);
        }

        double gunOffset = getGunHeadingRadians() - (Math.PI/2 - Math.atan2(p.y - getY(),p.x -  getX()));

        setTurnGunLeftRadians(NormaliseBearing(gunOffset));


    }


    //function to retrieve the state of robot
    private int getState()
    {
        int heading = Robo_States.getHeading(getHeading());
        int targetDistance = Robo_States.getEnemyDistance(target.distance);
        int targetBearing = Robo_States.getEnemyBearing(target.bearing);

        int state = Robo_States.MapState[targetDistance][targetBearing][heading][isHitWall][isHitByBullet];

        return state;
    }

    //normalizing bearing between the range -pi to pi
    double NormaliseBearing(double ang)
    {
        if (ang > PI)
            ang -= 2*PI;
        if (ang < -PI)
            ang += 2*PI;
        return ang;
    }

    //Distance between the two points on the coordinate plane
    public double getrange(double x1, double y1, double x2, double y2)
    {
        double xo = x2 - x1;
        double yo = y2 - y1;
        double h = Math.sqrt(xo * xo + yo * yo);
        return h;
    }

    //Event Methods

    public void onBulletHit(BulletHitEvent e)
    {
        if (target.name == e.getName())
        {
            reward += 2;

        }
    }

    //When the bullet misses hit another robot
    public void onBulletMissed(BulletMissedEvent e)
    {
        reward += -1;

    }

    //When the robot is hit by enemy's bullet
    public void onHitByBullet(HitByBulletEvent e)
    {
        reward += -2;
        isHitByBullet = 1;
    }

    //When the robot hits enemy robot
    public void onHitRobot(HitRobotEvent e)
    {
        reward += -2;
    }

    //When the robot hits the wall
    public void onHitWall(HitWallEvent e)
    {
        reward += -1;
        isHitWall = 1;
    }


    //When robot scans the enemy robot
    public void onScannedRobot(ScannedRobotEvent e)
    {
        if ((e.getDistance() < target.distance)||(target.name == e.getName()))
        {
            //Gets the absolute bearing to the point of the robot
            double absbearing_rad = (getHeadingRadians() + e.getBearingRadians()) % (2 * PI);

            //Storing all the information about the target robot
            target.name = e.getName();
            double h = NormaliseBearing(e.getHeadingRadians() - target.heading);
            h = h / (getTime() - target.ctime);
            target.changeHeading = h;
            target.x = getX() + Math.sin(absbearing_rad) * e.getDistance();
            target.y = getY() + Math.cos(absbearing_rad) * e.getDistance();
            target.bearing = e.getBearingRadians();
            target.heading = e.getHeadingRadians();
            target.ctime = getTime(); //game time at which this scan was produced
            target.speed = e.getVelocity();
            target.distance = e.getDistance();
            target.energy = e.getEnergy();
        }
    }

    public void onRobotDeath(RobotDeathEvent e)
    {
        if (e.getName() == target.name)
            target.distance = 10000;
    }

    //When robot wins the battle
    public void onWin(WinEvent event)
    {
        reward += 10;
        saveData();

        //Saving Battle History
        int winningFlag=1;

        PrintStream w = null;
        try {
            w = new PrintStream(new RobocodeFileOutputStream(getDataFile("battle.csv").getAbsolutePath(), true));
            w.println(reward+","+accu_reward+","+getRoundNum()+","+winningFlag+","+error_val);
            if (w.checkError())
                System.out.println("Could not save the data!");
            w.close();
        }
        catch (IOException e) {
            System.out.println("IOException trying to write: " + e);
        }
        finally {
            try {
                if (w != null)
                    w.close();
            }
            catch (Exception e) {
                System.out.println("Exception trying to close writer: " + e);
            }
        }


    }

    //On losing the battle robot dies
    public void onDeath(DeathEvent event)
    {

        reward += -5;
        saveData();

        //Saving Battle History
        int losingFlag=0;
        PrintStream w = null;
        try {
            w = new PrintStream(new RobocodeFileOutputStream(getDataFile("Q3_10.csv").getAbsolutePath(), true));
            w.println(reward+","+accu_reward+","+getRoundNum()+","+losingFlag+","+error_val);
            if (w.checkError())
                System.out.println("Could not save the data!");
            w.close();
        }
        catch (IOException e) {
            System.out.println("IOException trying to write: " + e);
        }
        finally {
            try {
                if (w != null)
                    w.close();
            }
            catch (Exception e) {
                System.out.println("Exception trying to close writer: " + e);
            }
        }
    }


    //============================================================
    //Load and save the data for the LUT
    //------------------------------------------------------------
    public void loadData()
    {
        try
        {
            table.loadData(getDataFile("LUT.dat"));
        }
        catch (Exception e)
        {
        }
    }

    public void saveData()
    {
        try
        {
            table.saveData(getDataFile("LUT.dat"));
        }
        catch (Exception e)
        {
            out.println("Exception trying to write: " + e);
        }
    }
}

