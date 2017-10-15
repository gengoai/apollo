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

package com.davidbracewell.apollo.stat.measure;

import com.davidbracewell.apollo.linear.NDArray;
import com.davidbracewell.guava.common.base.Preconditions;
import lombok.NonNull;
import org.apache.commons.math3.distribution.TDistribution;
import org.apache.commons.math3.util.FastMath;

/**
 * <p>Defines methodology to determine how related two items are.</p>
 *
 * @author David B. Bracewell
 */
public interface CorrelationMeasure extends SimilarityMeasure {

   @Override
   default double calculate(@NonNull NDArray v1, @NonNull NDArray v2) {
      Preconditions.checkArgument(v1.isVector() && v2.isVector(), "v1 and v2 must be bectors");
      v1.shape().checkDimensionMatch(v2.shape());
      return calculate(v1.toArray(), v2.toArray());
   }

   @Override
   default double calculate(ContingencyTable table) {
      throw new UnsupportedOperationException();
   }

   @Override
   double calculate(double[] v1, double[] v2);


   /**
    * Calculates the p-value for the correlation coefficient when N >= 6 using a one-tailed t-Test.
    *
    * @param r the correlation coefficient.
    * @param N the number of items
    * @return the non-directional p-value
    */
   default double pValue(double r, int N) {
      Preconditions.checkArgument(N >= 6, "N must be >= 6.");
      double t = (r * FastMath.sqrt(N - 2.0)) / FastMath.sqrt(1.0 - r * r);
      return 1.0 - new TDistribution(N - 2, 1)
                      .cumulativeProbability(t);
   }


}//END OF CorrelationMeasure
