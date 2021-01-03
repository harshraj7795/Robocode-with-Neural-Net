package RLNN_pkg;
import java.awt.geom.*;

public class Target {
    String name;
    public double bearing;
    public double heading;
    public long ctime;
    public double speed;
    public double x, y;
    public double distance;
    public double changeHeading;
    public double energy;

    //Calculating the new position of the enemy robot
    public Point2D.Double guessPosition(long when)
    {
        //ctime: when the scan data was produced
        //when: the time that we anticipate the bullet will reach the target
        //diff: is the difference between the two times
        double diff = when - ctime;
        double newY, newX;
        newX = x + Math.sin(heading) * speed * diff;
        newY = y + Math.cos(heading) * speed * diff;

        return new Point2D.Double(newX, newY);
    }

    public double guessX(long when)
    {
        long diff = when - ctime;
        System.out.println(diff);
        return x+Math.sin(heading)*speed*diff;
    }

    public double guessY(long when)
    {
        long diff = when - ctime;
        return y+Math.cos(heading)*speed*diff;
    }
}
