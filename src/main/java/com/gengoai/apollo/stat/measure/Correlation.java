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

package com.gengoai.apollo.stat.measure;

import com.gengoai.Lazy;
import com.gengoai.guava.common.base.Preconditions;
import lombok.NonNull;
import lombok.val;
import org.apache.commons.math3.stat.correlation.KendallsCorrelation;
import org.apache.commons.math3.stat.correlation.SpearmansCorrelation;
import org.apache.commons.math3.util.FastMath;

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
      @Override
      public double calculate(@NonNull double[] x, @NonNull double[] y) {
         Preconditions.checkArgument(x.length == y.length,
                                     "Dimension mismatch dim(x)=" + x.length + " != dim(y)=" + y.length);
         double x2 = 0d;
         double y2 = 0d;
         double sumX = 0d;
         double sumY = 0d;
         double xy = 0d;
         for (int i = 0; i < x.length; i++) {
            sumX += x[i];
            x2 += x[i] * x[i];
            sumY += y[i];
            y2 += y[i] * y[i];
            xy += x[i] * y[i];
         }
         double SSx = x2 - ((sumX * sumX) / x.length);
         double SSy = y2 - ((sumY * sumY) / x.length);
         double SSxy = xy - ((sumY * sumX) / x.length);
         return SSxy / Math.sqrt(SSx * SSy);
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
         return FastMath.pow(Pearson.calculate(v1, v2), 2d);
      }
   };

   public static void main(String[] args) {
      val x = new double[]{490, 500, 530, 550, 580, 590, 600, 600, 650, 700};
      val y = new double[]{560, 500, 510, 600, 600, 620, 550, 630, 650, 750};
      System.out.println(Pearson.calculate(x, y));
      System.out.println(Pearson.pValue(Pearson.calculate(x, y), x.length));
   }


}//END OF Correlation
