package com.davidbracewell.apollo.optimization;

/**
 * The interface Learning rate.
 *
 * @author David B. Bracewell
 */
public interface LearningRate {

   /**
    * Get double.
    *
    * @param currentLearningRate the current learning rate
    * @param time                the time
    * @param numProcessed        the num processed
    * @return the double
    */
   double get(double currentLearningRate, int time, int numProcessed);

   /**
    * Gets initial rate.
    *
    * @return the initial rate
    */
   double getInitialRate();

}//END OF LearningRate
