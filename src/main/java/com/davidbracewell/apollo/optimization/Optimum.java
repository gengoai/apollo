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

package com.davidbracewell.apollo.optimization;

import com.davidbracewell.function.SerializableToDoubleFunction;
import com.davidbracewell.stream.MStream;
import com.davidbracewell.tuple.Tuple2;
import lombok.NonNull;

import java.util.Comparator;
import java.util.Optional;
import java.util.stream.Stream;

/**
 * <p>Defines methodologies for comparing, testing, and selecting values and indexes based on optimizing values in a
 * specific direction, i.e. MAXIMUM or MINIMUM </p>
 *
 * @author David B. Bracewell
 */
public enum Optimum implements Comparator<Double> {
   /**
    * Optimizes for minimum values
    */
   MINIMUM {
      @Override
      public int compare(double v1, double v2) {
         return Double.compare(v1, v2);
      }

      @Override
      public boolean test(double value, double threshold) {
         return value <= threshold;
      }

      @Override
      public double startingValue() {
         return Double.POSITIVE_INFINITY;
      }

   },
   /**
    * Optimizes for maximum values
    */
   MAXIMUM {
      @Override
      public int compare(double v1, double v2) {
         return -Double.compare(v1, v2);
      }

      @Override
      public boolean test(double value, double threshold) {
         return value >= threshold;
      }

      @Override
      public double startingValue() {
         return Double.NEGATIVE_INFINITY;
      }
   };

   @Override
   public int compare(Double o1, Double o2) {
      if (o1 == null) return -1;
      if (o2 == null) return 1;
      return compare(o1, o2);
   }

   /**
    * Compares two values such that the most optimum value would result in being sorted to the front.
    *
    * @param v1 the first value
    * @param v2 the second value
    * @return the result of comparison (optimum value should be < 0)
    */
   public abstract int compare(double v1, double v2);

   /**
    * Select the best index and value from the given array.
    *
    * @param array the array to select from
    * @return the index and value of the best entry
    */
   public Tuple2<Integer, Double> optimum(@NonNull double[] array) {
      int bestIndex = optimumIndex(array);
      return Tuple2.of(bestIndex, array[bestIndex]);
   }

   public <T> Optional<T> optimum(@NonNull MStream<T> stream, @NonNull SerializableToDoubleFunction<T> function) {
      if (this == MAXIMUM) {
         return stream.max((t1, t2) -> Double.compare(function.applyAsDouble(t1), function.applyAsDouble(t2)));
      } else if (this == MINIMUM) {
         return stream.min((t1, t2) -> Double.compare(function.applyAsDouble(t1), function.applyAsDouble(t2)));
      }
      throw new IllegalStateException();
   }

   public <T> Optional<T> optimum(@NonNull Stream<T> stream, @NonNull SerializableToDoubleFunction<T> function) {
      if (this == MAXIMUM) {
         return stream.max(Comparator.comparingDouble(function::applyAsDouble));
      } else if (this == MINIMUM) {
         return stream.min(Comparator.comparingDouble(function::applyAsDouble));
      }
      throw new IllegalStateException();
   }

   /**
    * Selects the index of the best value in the given array
    *
    * @param array the array to select from
    * @return the index of the optimum value
    */
   public int optimumIndex(@NonNull double[] array) {
      double val = startingValue();
      int index = -1;
      for (int i = 0; i < array.length; i++) {
         if (test(array[i], val)) {
            val = array[i];
            index = i;
         }
      }
      return index;
   }

   /**
    * Selects the best value from the given array
    *
    * @param array the array to select from
    * @return the most optimum value
    */
   public double optimumValue(@NonNull double[] array) {
      double val = startingValue();
      for (double anArray : array) {
         if (test(anArray, val)) {
            val = anArray;
         }
      }
      return val;
   }

   /**
    * Gets a value that will also be the worst possible solution.
    *
    * @return the double
    */
   public abstract double startingValue();

   /**
    * Test if the given value passes the given threshold based on the direction of the optimum. For <code>MINIMUM</code>
    * this uses a less than equal to operator and for <code>MAXIMUM</code> this uses a greater than equal to operator.
    *
    * @param value     the value
    * @param threshold the threshold
    * @return the boolean
    */
   public abstract boolean test(double value, double threshold);

}//END OF Optimum
