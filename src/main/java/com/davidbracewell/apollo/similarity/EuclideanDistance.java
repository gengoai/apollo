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
import lombok.NonNull;

import java.util.Map;

/**
 * @author David B. Bracewell
 */
public class EuclideanDistance extends DistanceMeasure {

  private static final long serialVersionUID = 9055777643167317062L;
  private static transient DotProduct dotProduct = new DotProduct();

  @Override
  public double calculate(@NonNull Map<?, ? extends Number> m1, @NonNull Map<?, ? extends Number> m2) {
    double m1Sq = dotProduct.calculate(m1, m1);
    double m2Sq = dotProduct.calculate(m2, m2);
    double m12Sq = dotProduct.calculate(m1, m2);
    return Math.sqrt(m1Sq + m2Sq - 2 * m12Sq);
  }

  @Override
  public double calculate(@NonNull double[] v1, @NonNull double[] v2) {
    Preconditions.checkArgument(v1.length == v2.length, "Vector dimension mismatch");
    double m1Sq = dotProduct.calculate(v1, v1);
    double m2Sq = dotProduct.calculate(v2, v2);
    double m12Sq = dotProduct.calculate(v1, v2);
    return Math.sqrt(m1Sq + m2Sq - 2 * m12Sq);
  }

}//END OF EuclideanDistance
