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

import com.davidbracewell.EnhancedDoubleStatistics;
import com.davidbracewell.apollo.linalg.DenseVector;
import com.davidbracewell.apollo.linalg.Vector;
import com.davidbracewell.guava.common.base.Preconditions;
import lombok.NonNull;
import org.apache.commons.math3.stat.correlation.KendallsCorrelation;

import java.util.ArrayList;
import java.util.Collections;

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
      public double calculate(@NonNull double[] v1, @NonNull double[] v2) {
         Preconditions.checkArgument(v1.length == v2.length,
                                     "Vector k mismatch " + v1.length + " != " + v2.length);

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
         double den = Math.sqrt((v1Stats.getSumOfSquares() - Math.pow(v1Stats.getSum(),
                                                                      2) / n) * (v2Stats.getSumOfSquares() - Math.pow(
            v2Stats.getSum(), 2) / n));

         if (den == 0) {
            return 0;
         }

         return num / den;
      }
   },
   /**
    * <a href="https://en.wikipedia.org/wiki/Spearman%27s_rank_correlation_coefficient">Spearman's rank correlation
    * coefficient</a>
    */
   Spearman {
      @Override
      public double calculate(@NonNull double[] v1, @NonNull double[] v2) {
         Preconditions.checkArgument(v1.length == v2.length,
                                     "Vector k mismatch " + v1.length + " != " + v2.length);
         Preconditions.checkArgument(v1.length >= 2, "Need at least two elements");
         return Pearson.calculate(rank(v1), rank(v2));
      }
   },
   /**
    * <a href="https://en.wikipedia.org/wiki/Kendall_rank_correlation_coefficient"> Kendall's Tau-b rank correlation</a>
    */
   Kendall {
      final KendallsCorrelation kendallsCorrelation = new KendallsCorrelation();

      @Override
      public double calculate(double[] v1, double[] v2) {
         Preconditions.checkArgument(v1.length == v2.length,
                                     "Vector k mismatch " + v1.length + " != " + v2.length);
         Preconditions.checkArgument(v1.length >= 2, "Need at least two elements");
         return kendallsCorrelation.correlation(v1, v2);
      }
   },
   /**
    * <a href="https://en.wikipedia.org/wiki/Coefficient_of_determination">R^2 Coefficient of determination</a>
    */
   R_Squared {
      @Override
      public double calculate(double[] v1, double[] v2) {
         double r = Pearson.calculate(v1, v2);
         return r * r;
      }
   };


   private static double[] rank(double[] v) {
      ArrayList<RankPair> rankPairs = new ArrayList<>(v.length);
      for (int i = 0; i < v.length; i++) {
         rankPairs.add(new RankPair(i, v[i]));
      }
      rankPairs.trimToSize();
      Collections.sort(rankPairs);
      double[] out = new double[v.length];

      for (int i = 0; i < v.length; i++) {
         RankPair rp = rankPairs.get(i);
         double rank = i + 1;
         if (i > 0 && rp.value == rankPairs.get(i - 1).value) {
            rank = (rank + i) / 2d;
         } else if (i < v.length - 1 && rp.value == rankPairs.get(i + 1).value) {
            rank = (rank + i + 2) / 2d;
         }
         out[rp.index] = rank;
      }
      return out;
   }

   private static class RankPair implements Comparable<RankPair> {
      /**
       * The Index.
       */
      final int index;
      /**
       * The Value.
       */
      final double value;

      /**
       * Instantiates a new Rank pair.
       *
       * @param index the index
       * @param value the value
       */
      public RankPair(int index, double value) {
         this.index = index;
         this.value = value;
      }

      @Override
      public int compareTo(RankPair o) {
         return Double.compare(value, o.value);
      }
   }
}//END OF Correlation
