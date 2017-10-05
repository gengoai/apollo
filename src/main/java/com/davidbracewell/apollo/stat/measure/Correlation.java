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

import com.davidbracewell.Lazy;
import com.davidbracewell.guava.common.base.Preconditions;
import lombok.NonNull;
import org.apache.commons.math3.stat.correlation.KendallsCorrelation;
import org.apache.commons.math3.stat.correlation.PearsonsCorrelation;
import org.apache.commons.math3.stat.correlation.SpearmansCorrelation;
import org.apache.commons.math3.stat.regression.SimpleRegression;

/**
 * <p>Common methods for calculating the correlation between arrays of values.</p>
 *
 * @author David B. Bracewell
 */
public enum Correlation implements CorrelationMeasure {
   /**
    * <a href="https://en.wikipedia.org/wiki/Pearson_product-moment_correlation_coefficient">The Pearson product-moment
    * correlation coefficient.</a>
    */
   Pearson {
      final transient Lazy<PearsonsCorrelation> pearsonCorrelation = new Lazy<>(PearsonsCorrelation::new);

      @Override
      public double calculate(@NonNull double[] v1, @NonNull double[] v2) {
         return pearsonCorrelation.get().correlation(v1, v2);
      }
   },
   /**
    * <a href="https://en.wikipedia.org/wiki/Spearman%27s_rank_correlation_coefficient">Spearman's rank correlation
    * coefficient</a>
    */
   Spearman {
      final transient Lazy<SpearmansCorrelation> spearmansCorrelation = new Lazy<>(SpearmansCorrelation::new);

      @Override
      public double calculate(@NonNull double[] v1, @NonNull double[] v2) {
         Preconditions.checkArgument(v1.length == v2.length,
                                     "Vector dimension mismatch " + v1.length + " != " + v2.length);
         Preconditions.checkArgument(v1.length >= 2, "Need at least two elements");
         return spearmansCorrelation.get().correlation(v1, v2);
      }
   },
   /**
    * <a href="https://en.wikipedia.org/wiki/Kendall_rank_correlation_coefficient"> Kendall's Tau-b rank
    * correlation</a>
    */
   Kendall {
      final transient Lazy<KendallsCorrelation> kendallsCorrelation = new Lazy<>(KendallsCorrelation::new);

      @Override
      public double calculate(double[] v1, double[] v2) {
         return kendallsCorrelation.get().correlation(v1, v2);
      }
   },
   /**
    * <a href="https://en.wikipedia.org/wiki/Coefficient_of_determination">R^2 Coefficient of determination</a>
    */
   R_Squared {
      @Override
      public double calculate(double[] v1, double[] v2) {
         Preconditions.checkArgument(v1.length == v2.length,
                                     "Vector dimension mismatch " + v1.length + " != " + v2.length);
         SimpleRegression regression = new SimpleRegression();
         for (int i = 0; i < v1.length; ++i) {
            regression.addData(v1[i], v2[i]);
         }
         return regression.getRSquare();
      }
   }


}//END OF Correlation
