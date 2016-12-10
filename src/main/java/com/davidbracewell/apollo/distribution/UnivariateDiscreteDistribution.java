/*
 * (c) 2005 David B. Bracewell
 *
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package com.davidbracewell.apollo.distribution;

import com.davidbracewell.guava.common.base.Preconditions;

/**
 * A distribution backed whose density is a probability mass function of discrete a random variable <code>X</code>.
 *
 * @param <T> Convenience parameter to return instance type
 * @author David B. Bracewell
 */
public interface UnivariateDiscreteDistribution<T extends UnivariateDiscreteDistribution> extends UnivariateDistribution {

   /**
    * Draws one random sample from the distribution.
    *
    * @return the randomly drawn discrete random variable
    */
   int sample();

   /**
    * Draws <code>sampleSize</code> number random samples from the distribution.
    *
    * @param sampleSize the number of samples to take.
    * @return An array of <code>sampleSize</code> randomly drawn discrete random variables
    */
   default int[] sample(int sampleSize) {
      Preconditions.checkArgument(sampleSize > 0, "Size must be > 0");
      int[] samples = new int[sampleSize];
      for (int i = 0; i < sampleSize; i++) {
         samples[i] = sample();
      }
      return samples;
   }

   /**
    * Increments number of times the given variable has been observed by the given amount.
    *
    * @param variable             the variable to increment
    * @param numberOfObservations the number of times <code>variable</code> was observed.
    * @return this discrete distribution
    */
   T increment(int variable, int numberOfObservations);

   /**
    * Increments number of times the given variable has been observed by one.
    *
    * @param variable the variable to increment
    * @return this discrete distribution
    */
   default T increment(int variable) {
      return increment(variable, 1);
   }

   /**
    * Decrements number of times the given variable has been observed by the given amount.
    *
    * @param variable             the variable to decrement
    * @param numberOfObservations the number of observations of <code>variable</code> to remove
    * @return this discrete distribution
    */
   default T decrement(int variable, int numberOfObservations) {
      return increment(variable, -numberOfObservations);
   }

   /**
    * Decrements number of times the given variable has been observed by one.
    *
    * @param variable the variable to decrement
    * @return this discrete distribution
    */
   default T decrement(int variable) {
      return increment(variable, -1);
   }


}//END OF DiscreteDistribution
