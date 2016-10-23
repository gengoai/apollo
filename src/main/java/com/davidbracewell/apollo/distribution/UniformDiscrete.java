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

import com.google.common.base.Preconditions;
import lombok.NonNull;
import org.apache.commons.math3.random.RandomGenerator;
import org.apache.commons.math3.random.Well19937c;

import java.io.Serializable;

/**
 * The type Uniform discrete.
 *
 * @author David B. Bracewell
 */
public class UniformDiscrete implements DiscreteDistribution<UniformDiscrete>, Serializable {
   private static final long serialVersionUID = 1L;
   private int max;
   private int min;
   private final RandomGenerator random;


   /**
    * Instantiates a new Uniform discrete.
    *
    * @param k the k
    */
   public UniformDiscrete(int k) {
      this(0, k, new Well19937c());
   }

   /**
    * Instantiates a new Uniform discrete.
    *
    * @param min the min
    * @param max the max
    */
   public UniformDiscrete(int min, int max) {
      this(min, max, new Well19937c());
   }

   /**
    * Instantiates a new Uniform discrete.
    *
    * @param k      the k
    * @param random the random
    */
   public UniformDiscrete(int k, @NonNull RandomGenerator random) {
      this(0, k, random);
   }

   /**
    * Instantiates a new Uniform discrete.
    *
    * @param min    the min
    * @param max    the max
    * @param random the random
    */
   public UniformDiscrete(int min, int max, @NonNull RandomGenerator random) {
      Preconditions.checkArgument(min <= max, "Max must be >= min");
      this.min = min;
      this.max = max;
      this.random = random;
   }

   /**
    * Gets max.
    *
    * @return the max
    */
   public int getMax() {
      return max;
   }

   /**
    * Gets min.
    *
    * @return the min
    */
   public int getMin() {
      return min;
   }

   @Override
   public double getMode() {
      return Double.NaN;
   }

   @Override
   public double getMean() {
      return max / 2.0 + min / 2.0;
   }

   @Override
   public double getVariance() {
      return (range() * range()) / 12;
   }

   @Override
   public double probability(double value) {
      if (value < min || value > max) {
         return 0.0;
      }
      return 1.0 / range();
   }

   @Override
   public int sample() {
      return random.nextInt(max - min) + min;
   }

   @Override
   public UniformDiscrete increment(int variable, int amount) {
      return this;
   }

   @Override
   public double cumulativeProbability(double x) {
      if (x >= max) {
         return 1.0;
      } else if (x <= min) {
         return 0.0;
      }
      return (1.0 - min + x) / range();
   }

   private int range() {
      return max - min + 1;
   }

   @Override
   public double inverseCumulativeProbability(double p) {
      Preconditions.checkArgument(p >= 0 && p <= 1, "Invalid probability");
      if (p <= 0) {
         return min;
      } else if (p >= 1) {
         return max;
      }
      return Math.max(1, Math.ceil((range() * p) + min - 1));
   }


   /**
    * Sets max.
    *
    * @param max the max
    */
   public void setMax(int max) {
      Preconditions.checkArgument(min <= max, "Max must be >= min");
      this.max = max;
   }

   /**
    * Sets min.
    *
    * @param min the min
    */
   public void setMin(int min) {
      Preconditions.checkArgument(min <= max, "Max must be >= min");
      this.min = min;
   }
}//END OF UniformDiscrete
