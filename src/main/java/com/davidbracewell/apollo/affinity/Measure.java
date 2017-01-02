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

package com.davidbracewell.apollo.affinity;

import com.davidbracewell.apollo.analysis.Optimum;
import com.davidbracewell.apollo.linalg.DenseVector;
import com.davidbracewell.apollo.linalg.Vector;
import com.davidbracewell.apollo.linalg.VectorMap;
import com.davidbracewell.collection.counter.Counter;
import com.davidbracewell.guava.common.collect.Maps;
import lombok.NonNull;

import java.io.Serializable;
import java.util.Map;
import java.util.Set;

/**
 * <p>Calculates a metric between items, such as distance and similarity.</p>
 *
 * @author David B. Bracewell
 */
public interface Measure extends Serializable {

   /**
    * Calculate this measure using two double arrays as the input
    *
    * @param v1 the first double array
    * @param v2 the second double array
    * @return the metric result
    */
   default double calculate(@NonNull double[] v1, @NonNull double[] v2) {
      return calculate(DenseVector.wrap(v1), DenseVector.wrap(v2));
   }

   /**
    * Calculate this measure using two vectors as the input
    *
    * @param v1 the first vector
    * @param v2 the second vector
    * @return the metric result
    */
   default double calculate(@NonNull Vector v1, @NonNull Vector v2) {
      return calculate(VectorMap.wrap(v1), VectorMap.wrap(v2));
   }

   /**
    * Calculate this measure using two counters as the input
    *
    * @param c1 the first counter
    * @param c2 the second counter
    * @return the metric result
    */
   default double calculate(@NonNull Counter<?> c1, @NonNull Counter<?> c2) {
      return calculate(c1.asMap(), c2.asMap());
   }

   /**
    * Calculate this measure using two sets as the input
    *
    * @param c1 the first set
    * @param c2 the second set
    * @return the metric result
    */
   default double calculate(@NonNull Set<?> c1, @NonNull Set<?> c2) {
      return calculate(Maps.asMap(c1, d -> 1), Maps.asMap(c2, d -> 1));
   }

   /**
    * Calculate this measure using two maps as the input
    *
    * @param m1 the first map
    * @param m2 the second map
    * @return the metric result
    */
   double calculate(Map<?, ? extends Number> m1, Map<?, ? extends Number> m2);


   /**
    * Gets what kind of optimum should be used with this measure, i.e. is bigger or smaller better.
    *
    * @return the optimum
    */
   Optimum getOptimum();

}//END OF Measure
