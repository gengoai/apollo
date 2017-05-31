package com.davidbracewell.apollo.optimization;

/**
 * @author David B. Bracewell
 */
public interface LearningRate {

   double get(double currentLearningRate, int time, int numProcessed);

   double getInitialRate();

}//END OF LearningRate
