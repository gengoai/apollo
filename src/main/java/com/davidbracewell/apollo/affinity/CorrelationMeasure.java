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

import com.davidbracewell.apollo.linalg.Vector;
import com.davidbracewell.collection.HashMapIndex;
import com.davidbracewell.collection.Index;
import com.google.common.base.Preconditions;
import com.google.common.collect.Sets;
import lombok.NonNull;
import org.apache.commons.math3.distribution.TDistribution;

import java.util.Map;

/**
 * The interface Correlation measure.
 *
 * @author David B. Bracewell
 */
public interface CorrelationMeasure extends SimilarityMeasure {

  @Override
  default double calculate(@NonNull Map<?, ? extends Number> m1, @NonNull Map<?, ? extends Number> m2) {
    Index index = new HashMapIndex<>(Sets.union(m1.keySet(), m2.keySet()));
    double[] v1 = new double[index.size()];
    double[] v2 = new double[index.size()];
    for (int i = 0; i < index.size(); i++) {
      v1[i] = m1.containsKey(index.get(i)) ? m1.get(index.get(i)).doubleValue() : 0d;
      v2[i] = m2.containsKey(index.get(i)) ? m2.get(index.get(i)).doubleValue() : 0d;
    }
    return calculate(v1, v2);
  }

  @Override
  default double calculate(@NonNull Vector v1, @NonNull Vector v2) {
    Preconditions.checkArgument(v1.dimension() == v2.dimension(), "Vector dimension mismatch " + v1.dimension() + " != " + v2.dimension());
    return calculate(v1.toArray(), v2.toArray());
  }

  @Override
  double calculate(double[] v1, double[] v2);


  /**
   * Calculates the p-value for the correlation coefficient when N >= 6 using a t-Test.
   *
   * @param r the correlation coefficient.
   * @param N the number of items
   * @return the non-directional p-value
   */
  static double pValue(double r, int N) {
    Preconditions.checkArgument(N >= 6, "N must be >= 6.");
    double t = r / Math.sqrt((1 - r * r) / (N - 2));
    return 1.0 - new TDistribution(N - 2).cumulativeProbability(t);
  }

}//END OF CorrelationMeasure
