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

/**
 * The interface Discrete distribution.
 *
 * @param <T> the type parameter
 * @author David B. Bracewell
 */
public interface DiscreteDistribution<T extends DiscreteDistribution> extends Density {

   /**
    * Sample int.
    *
    * @return the int
    */
   int sample();

   /**
    * Sample int [ ].
    *
    * @param size the size
    * @return the int [ ]
    */
   default int[] sample(int size) {
      Preconditions.checkArgument(size > 0, "Size must be > 0");
      int[] samples = new int[size];
      for (int i = 0; i < size; i++) {
         samples[i] = sample();
      }
      return samples;
   }

   /**
    * Increment t.
    *
    * @param variable the k
    * @param amount   the value
    * @return the t
    */
   T increment(int variable, int amount);

   /**
    * Increment t.
    *
    * @param variable the k
    * @return the t
    */
   default T increment(int variable) {
      return increment(variable, 1);
   }

   /**
    * Decrement t.
    *
    * @param variable the k
    * @param amount   the value
    * @return the t
    */
   default T decrement(int variable, int amount) {
      return increment(variable, -amount);
   }

   /**
    * Decrement t.
    *
    * @param variable the k
    * @return the t
    */
   default T decrement(int variable) {
      return increment(variable, -1);
   }


}//END OF Distribution
