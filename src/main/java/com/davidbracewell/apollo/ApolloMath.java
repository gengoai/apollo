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
import com.davidbracewell.apollo.linalg.Vector;
import com.davidbracewell.tuple.Tuple2;
import lombok.NonNull;

import java.math.BigDecimal;
import java.util.stream.DoubleStream;
import java.util.stream.IntStream;

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
    * Add double.
    *
    * @param v1 the v 1
    * @param v2 the v 2
    * @return the double
    */
   static double add(double v1, double v2) {
      return v1 + v2;
   }

   /**
    * Add squared double.
    *
    * @param v1 the v 1
    * @param v2 the v 2
    * @return the double
    */
   static double addSquared(double v1, double v2) {
      return v1 * v1 + v2 * v2;
   }

   /**
    * Arg max tuple 2.
    *
    * @param array the array
    * @return the tuple 2
    */
   static Tuple2<Integer, Double> argMax(@NonNull double[] array) {
      int index = Optimum.MAXIMUM.selectBestIndex(array);
      return Tuple2.of(index, array[index]);
   }

   /**
    * Arg max tuple 2.
    *
    * @param vector the vector
    * @return the tuple 2
    */
   static Tuple2<Integer, Double> argMax(@NonNull Vector vector) {
      return argMax(vector.toArray());
   }

   /**
    * Arg min tuple 2.
    *
    * @param array the array
    * @return the tuple 2
    */
   static Tuple2<Integer, Double> argMin(@NonNull double... array) {
      int index = Optimum.MINIMUM.selectBestIndex(array);
      return Tuple2.of(index, array[index]);
   }

   /**
    * Arg min tuple 2.
    *
    * @param vector the vector
    * @return the tuple 2
    */
   static Tuple2<Integer, Double> argMin(@NonNull Vector vector) {
      return argMin(vector.toArray());
   }

   /**
    * Average double.
    *
    * @param array the array
    * @return the double
    */
   static double average(@NonNull double... array) {
      return DoubleStream.of(array).average().orElse(Double.NaN);
   }

   /**
    * Average double.
    *
    * @param array the array
    * @return the double
    */
   static double average(@NonNull int... array) {
      return IntStream.of(array).average().orElse(Double.NaN);
   }

   /**
    * Divide double.
    *
    * @param v1 the v 1
    * @param v2 the v 2
    * @return the double
    */
   static double divide(double v1, double v2) {
      return v1 / v2;
   }

   /**
    * Log 2 double.
    *
    * @param number the number
    * @return the double
    */
   static double log2(double number) {
      return Math.log(number) / LOG2;
   }

   /**
    * Log sum double.
    *
    * @param v the v
    * @return the double
    */
   static double logSum(@NonNull double... v) {
      double m = v[0];
      for (int i = 1; i < v.length; ++i) {
         m = Math.max(m, v[i]);
      }
      double s = 0.;
      for (int i = 0; i < v.length; ++i) {
         s += Math.exp(-(m - v[i]));
      }
      return m + Math.log(s);
   }

   /**
    * Multiply double.
    *
    * @param v1 the v 1
    * @param v2 the v 2
    * @return the double
    */
   static double multiply(double v1, double v2) {
      return v1 * v2;
   }

   /**
    * Subtract double.
    *
    * @param v1 the v 1
    * @param v2 the v 2
    * @return the double
    */
   static double subtract(double v1, double v2) {
      return v1 - v2;
   }

   /**
    * Sum double.
    *
    * @param array the array
    * @return the double
    */
   static double sum(@NonNull double... array) {
      return DoubleStream.of(array).sum();
   }

   /**
    * Sum double.
    *
    * @param array the array
    * @return the double
    */
   static double sum(@NonNull int... array) {
      return IntStream.of(array).sum();
   }

   /**
    * Truncate double.
    *
    * @param value     the value
    * @param precision the precision
    * @return the double
    */
   static double truncate(double value, int precision) {
      return BigDecimal.valueOf(value).setScale(precision, BigDecimal.ROUND_HALF_UP).doubleValue();
   }


}//END OF ApolloMath
