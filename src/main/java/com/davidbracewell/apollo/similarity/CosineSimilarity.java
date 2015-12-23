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

import com.davidbracewell.apollo.stats.ContingencyTable;
import com.google.common.base.Preconditions;
import lombok.NonNull;

import java.util.Map;

/**
 * @author David B. Bracewell
 */
public class CosineSimilarity extends SimilarityMeasure {

  private static final long serialVersionUID = -4081205412763789054L;

  @Override
  public double calculate(@NonNull Map<?, ? extends Number> m1, @NonNull Map<?, ? extends Number> m2) {
    double dp = 0;
    double mag1 = 0;
    double mag2 = 0;

    for (Object key : m1.keySet()) {
      mag1 += Math.pow(m1.get(key).doubleValue(), 2);
      if (m2.containsKey(key)) {
        dp += (m1.get(key).doubleValue() * m2.get(key).doubleValue());
      }
    }
    for (Object key : m2.keySet()) {
      mag2 += Math.pow(m2.get(key).doubleValue(), 2);
    }

    return dp / (Math.sqrt(mag1) * Math.sqrt(mag2));
  }

  @Override
  public double calculate(@NonNull ContingencyTable table) {
    Preconditions.checkArgument(table.rowCount() == table.columnCount() && table.rowCount() == 2, "Only supports 2x2 contingency tables.");
    return table.get(0, 0) / Math.sqrt(table.columnSum(0) + table.rowSum(0));
  }

  @Override
  public double calculate(@NonNull double[] v1, @NonNull double[] v2) {
    Preconditions.checkArgument(v1.length == v2.length, "Vector dimension mismatch " + v1.length + " != " + v2.length);
    double dp = 0d;
    double mag1 = 0d;
    double mag2 = 0d;
    for (int i = 0; i < v1.length; i++) {
      dp += (v1[i] * v2[i]);
      mag1 += Math.pow(v1[i], 2);
      mag2 += Math.pow(v2[i], 2);
    }
    return dp / (Math.sqrt(mag1) * Math.sqrt(mag2));
  }

}//END OF DotProduct
