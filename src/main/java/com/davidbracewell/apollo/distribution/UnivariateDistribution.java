package com.davidbracewell.apollo.distribution;

import com.google.common.base.Preconditions;

import java.io.Serializable;

/**
 * <p>Encapsulates a probability density function.</p>
 *
 * @author David B. Bracewell
 */
public interface UnivariateDistribution extends Serializable {

   /**
    * Calculates the probability, i.e. <code>pmf(x)</code>, of the given value
    *
    * @param x the value to calculate the probability of
    * @return the probability of <code>x</code>
    */
   double probability(double x);

   /**
    * Calculates the log probability of the given value
    *
    * @param x the value to calculate the log probability of
    * @return the log probability of <code>x</code>
    */
   default double logProbability(double x) {
      return Math.log(probability(x));
   }

   /**
    * Calculates the cumulative probability, which is the probability that the value in this density function is less
    * than or equal to the given value <code>x</code>.
    *
    * @param x the upper bound to use for calculating the cumulative probability
    * @return the cumulative probability
    */
   double cumulativeProbability(double x);

   /**
    * Calculates the cumulative probability, which is the probability that the value in this density function falls
    * between  the given <code>lowerBound</code> and <code>higherBound</code>.
    *
    * @param lowerBound  the lower bound
    * @param higherBound the higher bound
    * @return the cumulative probability
    */
   default double cumulativeProbability(double lowerBound, double higherBound) {
      Preconditions.checkArgument(lowerBound <= higherBound, "Higher bound must be >= lower Bound");
      return cumulativeProbability(higherBound) - cumulativeProbability(lowerBound);
   }

   /**
    * Calculates the inverse cumulative probability, which gives the value associated with the given cumulative
    * probability <code>p</code>.
    *
    * @param p the probability whose value we want to look up
    * @return the value at the given cumulative probability
    */
   double inverseCumulativeProbability(double p);

   /**
    * Gets the value in that results in the density function having its maximum value
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


}// END OF UnivariateDistribution
