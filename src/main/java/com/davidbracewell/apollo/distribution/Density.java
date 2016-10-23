package com.davidbracewell.apollo.distribution;

import com.davidbracewell.function.SerializableDoubleUnaryOperator;

/**
 * <p>Encapsulates a probability density function.</p>
 *
 * @author David B. Bracewell
 */
public interface Density extends SerializableDoubleUnaryOperator {

   /**
    * Log output of the density function.
    *
    * @param v the value to calculate density at
    * @return the then density
    */
   default double logApplyAsDouble(double v) {
      return Math.log(applyAsDouble(v));
   }

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
