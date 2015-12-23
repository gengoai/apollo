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

package com.davidbracewell.apollo.similarity;

import com.google.common.base.Preconditions;
import com.google.common.collect.Sets;
import lombok.NonNull;

import java.util.Map;

/**
 * @author David B. Bracewell
 */
public class HammingDistance extends DistanceMeasure {
  @Override
  public double calculate(@NonNull Map<?, ? extends Number> m1, @NonNull  Map<?, ? extends Number> m2) {
    double sum = 0;
    for (Object o : Sets.union(m1.keySet(), m2.keySet())) {
      double d1 = m1.containsKey(o) ? m1.get(o).doubleValue() : 0d;
      double d2 = m2.containsKey(o) ? m2.get(o).doubleValue() : 0d;
      if (d1 != d2) {
        sum++;
      }
    }
    return sum;
  }

  @Override
  public double calculate(@NonNull double[] v1, @NonNull double[] v2) {
    Preconditions.checkArgument(v1.length == v2.length, "Dimension mismatch " + v1.length + " != " + v2.length);
    double sum = 0;
    for (int i = 0; i < v1.length; i++) {
      if (v1[i] != v2[i]) {
        sum++;
      }
    }
    return sum;
  }
}//END OF HammingDistance
