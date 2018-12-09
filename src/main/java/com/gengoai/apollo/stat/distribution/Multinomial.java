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

package com.gengoai.apollo.stat.distribution;

import com.gengoai.Validation;
import org.apache.commons.math3.distribution.EnumeratedIntegerDistribution;
import org.apache.commons.math3.random.RandomGenerator;
import org.apache.commons.math3.random.Well19937c;

import java.io.Serializable;
import java.util.Arrays;
import java.util.stream.IntStream;

/**
 * A generalization of a {@link Binomial} distribution to <code>K</code> different choices.
 *
 * @author David B. Bracewell
 */
public class Multinomial implements UnivariateDiscreteDistribution<Multinomial>, Serializable {
   private static final long serialVersionUID = 1L;
   private final int[] values;
   private final double alpha;
   private final double alphaTimesV;
   private final RandomGenerator random;
   private int sum = 0;
   private volatile EnumeratedIntegerDistribution wrapped;

   /**
    * Instantiates a new Multinomial.
    *
    * @param k      the number of possible values the random variable can take
    * @param alpha  the smoothing parameter
    * @param random the random number generator to use for sampling
    */
   public Multinomial(int k, double alpha, RandomGenerator random) {
      Validation.checkArgument(k > 0, "Size must be > 0");
      Validation.checkArgument(Double.isFinite(alpha), "Alpha must be finite");
      Validation.checkArgument(alpha > 0, "Alpha must be > 0");
      this.values = new int[k];
      this.alpha = alpha;
      this.alphaTimesV = alpha * k;
      this.random = random;
   }

   /**
    * Instantiates a new Multinomial.
    *
    * @param k the number of possible values the random variable can take
    */
   public Multinomial(int k) {
      this(k, 0, new Well19937c());
   }

   /**
    * Instantiates a new Multinomial.
    *
    * @param k     the number of possible values the random variable can take
    * @param alpha the smoothing parameter
    */
   public Multinomial(int k, double alpha) {
      this(k, alpha, new Well19937c());
   }

   @Override
   public double cumulativeProbability(double x) {
      return getDistribution().cumulativeProbability((int) x);
   }

   @Override
   public double cumulativeProbability(double lowerBound, double higherBound) {
      return getDistribution().cumulativeProbability((int) lowerBound, (int) higherBound);
   }

   private EnumeratedIntegerDistribution getDistribution() {
      if (wrapped == null) {
         synchronized (this) {
            if (wrapped == null) {
               EnumeratedIntegerDistribution eid =
                  new EnumeratedIntegerDistribution(random, IntStream.range(0, values.length).toArray(),
                                                    IntStream.range(0, values.length)
                                                             .mapToDouble(this::probability)
                                                             .toArray()
                  );
               this.wrapped = eid;
               return eid;
            }
         }
      }
      return wrapped;
   }

   @Override
   public double getMean() {
      return getDistribution().getNumericalMean();
   }

   @Override
   public double getMode() {
      int max = values[0];
      int maxi = 0;
      for (int i = 1; i < values.length; i++) {
         if (values[i] > max) {
            max = values[i];
            maxi = i;
         }
      }
      return maxi;
   }

   /**
    * Gets the total number of observations
    *
    * @return the total number of observations
    */
   public double getTotalObservations() {
      return sum;
   }

   @Override
   public double getVariance() {
      return getDistribution().getNumericalVariance();
   }

   @Override
   public Multinomial increment(int variable, int numberOfObservations) {
      this.values[variable] += numberOfObservations;
      this.sum += numberOfObservations;
      this.wrapped = null;
      return this;
   }

   @Override
   public double inverseCumulativeProbability(double p) {
      return getDistribution().inverseCumulativeProbability((int) p);
   }

   @Override
   public double probability(double x) {
      if (x < 0 || x >= values.length) {
         return 0.0;
      }
      return (values[(int) x] + alpha) / (sum + alphaTimesV);
   }

   @Override
   public int sample() {
      return getDistribution().sample();
   }

   @Override
   public String toString() {
      return Arrays.toString(values);
   }

}//END OF Multinomial
