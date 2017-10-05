package com.davidbracewell.apollo.stat.distribution;

import com.davidbracewell.guava.common.base.Preconditions;

/**
 * The interface Real distribution.
 *
 * @param <T> the type parameter
 * @author David B. Bracewell
 */
public interface UnivariateRealDistribution<T extends UnivariateRealDistribution> extends UnivariateDistribution {

   /**
    * Adds the given observed value as part of the distribution
    *
    * @param value the observed value
    * @return this distribution
    */
   T addValue(double value);

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
    * Draws one random values from the distribution.
    *
    * @return the randomly drawn real value
    */
   double sample();


}//END OF RealDistribution
