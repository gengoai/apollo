package com.davidbracewell.apollo.optimization;

/**
 * @author David B. Bracewell
 */
public interface LearningRate {

   double get(int time, int numProcessed);

}//END OF LearningRate
