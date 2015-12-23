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

import com.davidbracewell.apollo.linalg.DenseVector;
import com.davidbracewell.apollo.linalg.Vector;
import com.davidbracewell.collection.EnhancedDoubleStatistics;
import com.google.common.base.Preconditions;

/**
 * @author David B. Bracewell
 */
public class PearsonCorrelation extends AbstractCorrelationSimilarity {

  private static final long serialVersionUID = -2283302761286695056L;

  @Override
  public double calculate(double[] v1, double[] v2) {
    Preconditions.checkNotNull(v1, "Vectors cannot be null");
    Preconditions.checkNotNull(v2, "Vectors cannot be null");
    Preconditions.checkArgument(v1.length == v2.length, "Vector dimension mismatch " + v1.length + " != " + v2.length);

    Vector dv1 = DenseVector.wrap(v1);
    Vector dv2 = DenseVector.wrap(v2);

    double n = dv1.dimension();
    double dot = dv1.dot(dv2);

    if (dot == 0) {
      return 0;
    }

    EnhancedDoubleStatistics v1Stats = dv1.statistics();
    EnhancedDoubleStatistics v2Stats = dv2.statistics();

    double num = dot - (v1Stats.getSum() * v2Stats.getSum() / n);
    double den = Math.sqrt((v1Stats.getSumOfSquares() - Math.pow(v1Stats.getSum(), 2) / n) * (v2Stats.getSumOfSquares() - Math.pow(v2Stats.getSum(), 2) / n));

    if (den == 0) {
      return 0;
    }

    return num / den;
  }

}//END OF PearsonCorrelation
