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

/**
 * @author David B. Bracewell
 */
public class SpearmanCorrelation extends AbstractCorrelationSimilarity {

  private static final long serialVersionUID = 4729182287837932621L;

  @Override
  public double calculate(double[] v1, double[] v2) {
    Preconditions.checkNotNull(v1, "Vectors cannot be null");
    Preconditions.checkNotNull(v2, "Vectors cannot be null");
    Preconditions.checkArgument(v1.length == v2.length, "Vector dimension mismatch " + v1.length + " != " + v2.length);
    return new org.apache.commons.math3.stat.correlation.SpearmansCorrelation().correlation(v1, v2);
  }

}//END OF SpearmanCorrelation
