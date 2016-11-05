package com.davidbracewell.apollo.distribution;

import com.google.common.base.Preconditions;

/**
 * The interface Real distribution.
 *
 * @param <T> the type parameter
 * @author David B. Bracewell
 */
public interface UnivariateRealDistribution<T extends UnivariateRealDistribution> extends UnivariateDistribution {

   /**
    * Draws one random values from the distribution.
    *
    * @return the randomly drawn real value
    */
   double sample();

   /**
    * Draws <code>sampleSize</code> number random values from the distribution.
    *
    * @param sampleSize the number of samples to take.
    * @return An array of <code>sampleSize</code> randomly drawn values
    */
   default double[] sample(int sampleSize) {
      Preconditions.checkArgument(sampleSize > 0, "Size must be > 0");
      double[] samples = new double[sampleSize];
      for (int i = 0; i < sampleSize; i++) {
         samples[i] = sample();
      }
      return samples;
   }


   /**
    * Adds the given observed value as part of the distribution
    *
    * @param value the observed value
    * @return this distribution
    */
   T addValue(double value);


}//END OF RealDistribution
