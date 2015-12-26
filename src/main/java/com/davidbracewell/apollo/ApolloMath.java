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

import com.davidbracewell.apollo.linalg.DenseVector;
import com.davidbracewell.apollo.linalg.Vector;
import com.davidbracewell.collection.Collect;
import com.davidbracewell.tuple.Tuple2;
import lombok.NonNull;

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
   * Log 2 double.
   *
   * @param number the number
   * @return the double
   */
  static double log2(double number) {
    return Math.log(number) / LOG2;
  }


  /**
   * Average double.
   *
   * @param array the array
   * @return the double
   */
  static double average(@NonNull double[] array) {
    return DoubleStream.of(array).average().orElse(Double.NaN);
  }

  /**
   * Sum double.
   *
   * @param array the array
   * @return the double
   */
  static double sum(@NonNull double[] array) {
    return DoubleStream.of(array).sum();
  }

  /**
   * Sum double.
   *
   * @param array the array
   * @return the double
   */
  static double sum(@NonNull int[] array) {
    return IntStream.of(array).sum();
  }

  /**
   * Average double.
   *
   * @param array the array
   * @return the double
   */
  static double average(@NonNull int[] array) {
    return IntStream.of(array).average().orElse(Double.NaN);
  }


  /**
   * Arg max tuple 2.
   *
   * @param array the array
   * @return the tuple 2
   */
  static Tuple2<Integer, Double> argMax(@NonNull double[] array) {
    return argMax(DenseVector.wrap(array));
  }

  /**
   * Arg max tuple 2.
   *
   * @param vector the vector
   * @return the tuple 2
   */
  static Tuple2<Integer, Double> argMax(@NonNull Vector vector) {
    int maxI = -1;
    double maxV = Double.NEGATIVE_INFINITY;
    for (Vector.Entry entry : Collect.asIterable(vector.nonZeroIterator())) {
      if (entry.getValue() > maxV) {
        maxV = entry.getValue();
        maxI = entry.getIndex();
      }
    }
    return Tuple2.of(maxI, maxV);
  }

  /**
   * Arg min tuple 2.
   *
   * @param array the array
   * @return the tuple 2
   */
  static Tuple2<Integer, Double> argMin(@NonNull double[] array) {
    return argMin(DenseVector.wrap(array));
  }

  /**
   * Arg min tuple 2.
   *
   * @param vector the vector
   * @return the tuple 2
   */
  static Tuple2<Integer, Double> argMin(@NonNull Vector vector) {
    int minI = -1;
    double minV = Double.POSITIVE_INFINITY;
    for (Vector.Entry entry : Collect.asIterable(vector.nonZeroIterator())) {
      if (entry.getValue() < minV) {
        minV = entry.getValue();
        minI = entry.getIndex();
      }
    }
    return Tuple2.of(minI, minV);
  }

}//END OF ApolloMath