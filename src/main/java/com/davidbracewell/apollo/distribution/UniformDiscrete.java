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
   private final int k;
   private final RandomGenerator random;


   /**
    * Instantiates a new Uniform discrete.
    *
    * @param k the k
    */
   public UniformDiscrete(int k) {
      this(k, new Well19937c());
   }

   /**
    * Instantiates a new Uniform discrete.
    *
    * @param k      the k
    * @param random the random
    */
   public UniformDiscrete(int k, @NonNull RandomGenerator random) {
      Preconditions.checkArgument(k > 0, "K must be > 0");
      this.k = k;
      this.random = random;
   }

   @Override
   public double probability(int value) {
      if (value < 0 || value >= k) {
         return 0.0;
      }
      return 1.0 / k;
   }

   @Override
   public int sample() {
      return random.nextInt(k);
   }

   @Override
   public UniformDiscrete increment(int variable, int amount) {
      return this;
   }

   @Override
   public double cumulativeProbability(int x) {
      if (x < 0) {
         return 0;
      } else if (x >= k) {
         return 1.0;
      }
      return (double) x / (double) k;
   }

   @Override
   public double cumulativeProbability(int lowerBound, int higherBound) {
      if (higherBound - lowerBound <= 0) {
         return 0.0;
      } else if (lowerBound == 0 && higherBound >= k) {
         return 1.0;
      }
      return (higherBound - lowerBound) / k;
   }
}//END OF UniformDiscrete
