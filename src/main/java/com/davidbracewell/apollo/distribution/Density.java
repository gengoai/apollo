package com.davidbracewell.apollo.distribution;

import com.google.common.base.Preconditions;

import java.io.Serializable;

/**
 * <p>Encapsulates a probability density function.</p>
 *
 * @author David B. Bracewell
 */
public interface Density extends Serializable {

   /**
    * Probability double.
    *
    * @param value the value
    * @return the double
    */
   double probability(double value);


   /**
    * Log probability double.
    *
    * @param value the value
    * @return the double
    */
   default double logProbability(double value) {
      return Math.log(probability(value));
   }

   /**
    * Cumulative probability double.
    *
    * @param x the x
    * @return the double
    */
   double cumulativeProbability(double x);

   /**
    * Cumulative probability double.
    *
    * @param lowerBound  the lower bound
    * @param higherBound the higher bound
    * @return the double
    */
   default double cumulativeProbability(double lowerBound, double higherBound) {
      Preconditions.checkArgument(lowerBound <= higherBound, "Higher bound must be >= lower Bound");
      return cumulativeProbability(higherBound) - cumulativeProbability(lowerBound);
   }

   /**
    * Inverse cumulative probability double.
    *
    * @param p the p
    * @return the double
    */
   double inverseCumulativeProbability(double p);

   /**
    * Gets the mode of density function
    *
    * @return the mode
    */
   double getMode();

   /**
    * Gets the mean of the density function.
    *
    * @return the mean
    */
   double getMean();

   /**
    * Gets the variance of the density function.
    *
    * @return the variance
    */
   double getVariance();


}// END OF Density
