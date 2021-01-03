package RLNN_pkg;

public class Robo_States {
    public static int Num;
    public static int NumDistance=10;
    public static int NumBearing=4;
    public static int NumHeading=4;
    public static int NumHitWall=2;
    public static int NumHitByBullet=2;

    public static int MapState[][][][][];

    static
    {   //mapping the states
        MapState=new int[NumDistance][NumBearing][NumHeading][NumHitWall][NumHitByBullet];
        int count=0;
        for(int i=0;i<NumDistance;i++)
            for(int j=0;j<NumBearing;j++)
                for (int m=0;m<NumHeading;m++)
                    for(int p=0;p<NumHitWall;p++)
                        for(int q=0;q<NumHitByBullet;q++)

                        {
                            MapState[i][j][m][p][q]=count++;

                        }
        Num=count;
    }

    //getting enemy distance
    public static int getEnemyDistance(double value)
    {
        int distance=(int)(value/100);
        if(distance>NumDistance-1)
            distance=NumDistance-1;
        return distance;

    }

    //getting enemy bearing
    public static int getEnemyBearing(double bearing)
    {
        double  totalAngle=Math.PI*2;
        if(bearing<0)
            bearing=totalAngle+bearing;

        double angle=totalAngle/NumBearing;
        double newBearing=bearing+angle/2;
        if(newBearing>totalAngle)
            newBearing=newBearing-totalAngle;
        return (int) (newBearing/angle);
    }
    //getting enemy heading
    public static int getHeading(double heading)
    {
        double totalAngle=360.0d;
        double angle=totalAngle/NumHeading;
        double newHeading=heading+angle/2;
        if(newHeading>totalAngle)
            newHeading-=totalAngle;
        return(int)(newHeading/angle);
    }

    //function for getting state from index
    public static int[] getStateFromIndex(int index)
    {
        int distance = index/(NumBearing*NumHeading*NumHitWall*NumHitByBullet);
        int remain = index % (NumBearing*NumHeading*NumHitWall*NumHitByBullet);
        int bearing = remain/(NumHeading*NumHitWall*NumHitByBullet);
        remain = remain % (NumHeading*NumHitWall*NumHitByBullet);
        int head = remain/(NumHitWall*NumHitByBullet);
        remain = remain % (NumHitWall*NumHitByBullet);
        int hitwall = remain/(NumHitByBullet);
        int hitbybullet = remain % (NumHitByBullet);

        int[] states = new int[5];
        states[0]=distance;
        states[1]=bearing;
        states[2]=head;
        states[3]=hitwall;
        states[4]=hitbybullet;
        return states;
    }

}
