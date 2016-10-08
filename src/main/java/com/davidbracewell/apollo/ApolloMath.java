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

package com.davidbracewell.apollo;

import com.davidbracewell.apollo.affinity.Optimum;
import com.davidbracewell.tuple.Tuple2;
import lombok.NonNull;

/**
 * The interface Apollo math.
 *
 * @author David B. Bracewell
 */
public interface ApolloMath {

   /**
    * The constant LOG2.
    */
   double LOG2 = Math.log(2);


   /**
    * Adds the squared values of two doubles (useful as a method reference)
    *
    * @param v1 value 1
    * @param v2 value 2
    * @return the sum of value 1 squared and value 2 squared
    */
   static double addSquared(double v1, double v2) {
      return v1 * v1 + v2 * v2;
   }

   /**
    * Determines the maximum value and index of that value in the given double array
    *
    * @param array the array whose maximum value we are calculating
    * @return a tuple of index and maximum value
    */
   static Tuple2<Integer, Double> argMax(@NonNull double[] array) {
      int index = Optimum.MAXIMUM.selectBestIndex(array);
      return Tuple2.of(index, array[index]);
   }

   /**
    * Determines the minimum value and index of that value in the given double array
    *
    * @param array the array whose minimum value we are calculating
    * @return a tuple of index and minimum value
    */
   static Tuple2<Integer, Double> argMin(@NonNull double... array) {
      int index = Optimum.MINIMUM.selectBestIndex(array);
      return Tuple2.of(index, array[index]);
   }

   /**
    * Calculates the base 2 log of a given number
    *
    * @param number the number to calculate the base 2 log of
    * @return the base 2 log of the given number
    */
   static double log2(double number) {
      return Math.log(number) / LOG2;
   }


}//END OF ApolloMath
