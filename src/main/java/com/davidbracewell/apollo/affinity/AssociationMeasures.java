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

import com.davidbracewell.apollo.distribution.NormalDistribution;
import com.davidbracewell.guava.common.base.Preconditions;
import com.davidbracewell.guava.common.math.DoubleMath;
import com.davidbracewell.guava.common.primitives.Doubles;
import lombok.NonNull;
import org.apache.commons.math3.distribution.ChiSquaredDistribution;
import org.apache.commons.math3.distribution.TDistribution;

/**
 * Common measures to determine the association, or dependence, of variables in a contingency table.
 *
 * @author David B. Bracewell
 */
public enum AssociationMeasures implements ContingencyTableCalculator {
   /**
    * Measures based on Mikolov et. al's "Distributed Representations of Words and Phrases and their Compositionality"
    */
   Mikolov {
      @Override
      public double calculate(ContingencyTable table) {
         Preconditions.checkArgument(table.rowCount() == table.columnCount() && table.rowCount() == 2,
                                     "Only supports 2x2 contingency tables.");
         double cooc = table.get(0, 0);
         double w1Count = table.get(0, 1);
         double w2Count = table.get(1, 0);
         double minCount = Math.min(w1Count, w2Count);
         double score = (cooc - minCount) / (w1Count * w2Count);
         if (Double.isFinite(score)) {
            return score;
         }
         return Double.MAX_VALUE;
      }
   },
   /**
    * <a href="https://en.wikipedia.org/wiki/Mutual_information">Mutual Information</a>
    */
   MI {
      @Override
      public double calculate(@NonNull ContingencyTable table) {
         double sum = 0d;
         for (int row = 0; row < table.rowCount(); row++) {
            for (int col = 0; col < table.columnCount(); col++) {
               sum += table.get(row, col) / table.getSum() * DoubleMath.log2(
                  table.get(row, col) / table.getExpected(row, col));
            }
         }
         return Doubles.isFinite(sum) ? sum : 0d;
      }
   },
   /**
    * <a href="https://en.wikipedia.org/wiki/Pointwise_mutual_information">Pointwise Mutual Information</a>
    */
   PMI {
      @Override
      public double calculate(@NonNull ContingencyTable table) {
         Preconditions.checkArgument(table.rowCount() == table.columnCount() && table.rowCount() == 2,
                                     "Only supports 2x2 contingency tables.");
         return DoubleMath.log2(table.get(0, 0)) - DoubleMath.log2(table.getExpected(0, 0));
      }
   },
   /**
    * <a href="https://en.wikipedia.org/wiki/Odds_ratio">Odds Ratio</a>
    */
   ODDS_RATIO {
      @Override
      public double calculate(@NonNull ContingencyTable table) {
         Preconditions.checkArgument(table.rowCount() == table.columnCount() && table.rowCount() == 2,
                                     "Only supports 2x2 contingency tables.");
         double v1 = table.get(0, 0) / table.get(0, 1);
         double v2 = table.get(1, 0) / table.get(1, 1);
         return v1 / v2;
      }

      @Override
      public double pValue(@NonNull ContingencyTable table) {
         NormalDistribution distribution = new NormalDistribution(0, 1);
         return 1.0 - distribution.cumulativeProbability(Math.log(calculate(table)));
      }
   },
   /**
    * <a href="https://en.wikipedia.org/wiki/Standard_score#T-score">T-Score, which is a standard score with mean of 50
    * and standard deviation of 10</a>
    */
   T_SCORE {
      @Override
      public double calculate(@NonNull ContingencyTable table) {
         Preconditions.checkArgument(table.rowCount() == table.columnCount() && table.rowCount() == 2,
                                     "Only supports 2x2 contingency tables.");
         return (table.get(0, 0) - table.getExpected(0, 0)) / Math.sqrt(table.get(0, 0));
      }

      @Override
      public double pValue(@NonNull ContingencyTable table) {
         TDistribution distribution = new TDistribution(table.degreesOfFreedom());
         return 1.0 - distribution.cumulativeProbability(calculate(table));
      }

   },
   /**
    * <a href="https://en.wikipedia.org/wiki/Pointwise_mutual_information#Normalized_pointwise_mutual_information_.28npmi.29">Normalized
    * Pointwise Mutual Information</a>
    */
   NPMI {
      @Override
      public double calculate(@NonNull ContingencyTable table) {
         Preconditions.checkArgument(table.rowCount() == table.columnCount() && table.rowCount() == 2,
                                     "Only supports 2x2 contingency tables.");
         if (table.get(0, 0) == 0) {
            return -1;
         }
         return DoubleMath.log2(table.get(0, 0) / table.getExpected(0, 0)) /
                   -DoubleMath.log2(table.get(0, 0) / table.getSum());
      }
   },
   /**
    * Approximation to the Poisson Stirling likelihood.
    */
   POISSON_STIRLING {
      @Override
      public double calculate(@NonNull ContingencyTable table) {
         Preconditions.checkArgument(table.rowCount() == table.columnCount() && table.rowCount() == 2,
                                     "Only supports 2x2 contingency tables.");
         return table.get(0, 0) * (Math.log(table.get(0, 0) / table.getExpected(0, 0)) - 1);
      }
   },
   /**
    * <a href="https://en.wikipedia.org/wiki/Chi-squared_test">x2 score</a>
    */
   CHI_SQUARE {
      @Override
      public double calculate(@NonNull ContingencyTable table) {
         double sumSq = 0d;
         for (int row = 0; row < table.rowCount(); row++) {
            for (int col = 0; col < table.columnCount(); col++) {
               double expected = table.getExpected(row, col);
               sumSq += Math.pow(table.get(row, col) - expected, 2) / expected;
            }
         }
         return Doubles.isFinite(sumSq) ? sumSq : 0d;
      }

      @Override
      public double pValue(@NonNull ContingencyTable table) {
         ChiSquaredDistribution distribution = new ChiSquaredDistribution(table.degreesOfFreedom());
         return 1.0 - distribution.cumulativeProbability(calculate(table));
      }
   },
   /**
    * <a href="https://en.wikipedia.org/wiki/G-test">g^2 or log-likelihood</a>
    */
   G_SQUARE {
      @Override
      public double calculate(@NonNull ContingencyTable table) {
         double sum = 0d;
         for (int row = 0; row < table.rowCount(); row++) {
            for (int col = 0; col < table.columnCount(); col++) {
               sum += table.get(row, col) * Math.log(table.get(row, col) / table.getExpected(row, col));
            }
         }
         return Doubles.isFinite(sum) ? 2 * sum : 0d;
      }

      @Override
      public double pValue(@NonNull ContingencyTable table) {
         ChiSquaredDistribution distribution = new ChiSquaredDistribution(table.degreesOfFreedom());
         return 1.0 - distribution.cumulativeProbability(calculate(table));
      }
   },
   /**
    * <a href="https://en.wikipedia.org/wiki/Relative_risk">Relative Risk</a>
    */
   RELATIVE_RISK {
      @Override
      public double calculate(@NonNull ContingencyTable table) {
         Preconditions.checkArgument(table.rowCount() == table.columnCount() && table.rowCount() == 2,
                                     "Only supports 2x2 contingency tables.");
         double v1 = table.get(0, 0) / table.rowSum(0);
         double v2 = table.get(1, 0) / table.rowSum(1);
         return v1 / v2;
      }

      @Override
      public double pValue(@NonNull ContingencyTable table) {
         NormalDistribution distribution = new NormalDistribution(0, 1);
         return 1.0 - distribution.cumulativeProbability(Math.log(calculate(table)));
      }
   }

}//END OF AssociationMeasures
