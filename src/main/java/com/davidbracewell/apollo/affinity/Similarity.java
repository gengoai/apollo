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

import com.davidbracewell.Math2;
import com.davidbracewell.guava.common.base.Preconditions;
import lombok.NonNull;

import java.util.Map;

/**
 * <p>Common methods for determining the similarity between two items</p>
 *
 * @author David B. Bracewell
 */
public enum Similarity implements SimilarityMeasure {
   /**
    * <a href="https://en.wikipedia.org/wiki/Dot_product">The dot product</a>
    */
   DotProduct {
      @Override
      public double calculate(@NonNull Map<?, ? extends Number> m1, Map<?, ? extends Number> m2) {
         double dp = 0;
         for (Object key : m1.keySet()) {
            if (m2.containsKey(key)) {
               dp += (m1.get(key).doubleValue() * m2.get(key).doubleValue());
            }
         }
         return dp;
      }
   },
   /**
    * <a href="https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient">The Dice Coefficient</a>
    */
   Dice {
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

         return 2 * dp / (mag1 + mag2);
      }

      @Override
      public double calculate(@NonNull ContingencyTable table) {
         Preconditions.checkArgument(table.rowCount() == table.columnCount() && table.rowCount() == 2,
                                     "Only supports 2x2 contingency tables.");
         return 2 * table.get(0, 0) / (table.columnSum(0) + table.rowSum(0));
      }
   },
   /**
    * <a href="http://wortschatz.uni-leipzig.de/~sbordag/aalw05/Referate/14_MiningRelationen_WSA_opt/Curran_03.pdf">Variation
    * of the Dice's Coefficient</a>
    */
   DiceGen2 {
      @Override
      public double calculate(@NonNull Map<?, ? extends Number> m1, @NonNull Map<?, ? extends Number> m2) {
         double dp = 0;
         double sum = 0;

         Number ov;
         for (Map.Entry<?, ? extends Number> e : m1.entrySet()) {
            ov = m2.get(e.getKey());
            if (ov != null) {
               dp += e.getValue().doubleValue() * ov.doubleValue();
            }
            sum += e.getValue().doubleValue();
         }

         for (Map.Entry<?, ? extends Number> e : m2.entrySet()) {
            sum += e.getValue().doubleValue();
         }

         return sum == 0 ? 0.0 : dp / sum;
      }

      @Override
      public double calculate(@NonNull ContingencyTable table) {
         Preconditions.checkArgument(table.rowCount() == table.columnCount() && table.rowCount() == 2,
                                     "Only supports 2x2 contingency tables.");
         return 2 * table.get(0, 0) / (table.columnSum(0) + table.rowSum(0));
      }
   },
   /**
    * <a href="https://en.wikipedia.org/wiki/Cosine_similarity">Cosine Similarity</a>
    */
   Cosine {
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
         Preconditions.checkArgument(table.rowCount() == table.columnCount() && table.rowCount() == 2,
                                     "Only supports 2x2 contingency tables.");
         return table.get(0, 0) / Math.sqrt(table.columnSum(0) + table.rowSum(0));
      }
   },
   /**
    * <a href="https://en.wikipedia.org/wiki/Jaccard_index">Jaccard Index</a>
    */
   Jaccard {
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

         return dp / ((mag1 + mag2) - dp);
      }

      @Override
      public double calculate(@NonNull ContingencyTable table) {
         Preconditions.checkArgument(table.rowCount() == table.columnCount() && table.rowCount() == 2,
                                     "Only supports 2x2 contingency tables.");
         return table.get(0, 0) / (table.get(0, 0) + table.get(0, 1) + table.get(1, 0));
      }
   },
   /**
    * <a href="https://en.wikipedia.org/wiki/Overlap_coefficient">Overlap Coefficient</a>
    */
   Overlap {
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

         return Math2.clip(dp / Math.min(mag1, mag2), -1, 1);
      }

      @Override
      public double calculate(@NonNull ContingencyTable table) {
         Preconditions.checkNotNull(table);
         Preconditions.checkArgument(table.rowCount() == table.columnCount() && table.rowCount() == 2,
                                     "Only supports 2x2 contingency tables.");
         return table.get(0, 0) / Math.min(table.columnSum(0), table.rowSum(0));
      }
   },
   /**
    * <a href="https://en.wikipedia.org/wiki/Cosine_similarity#Angular_distance_and_similarity">Angular similarity
    * (variant of Cosine)</a>
    */
   Angular {
      @Override
      public double calculate(@NonNull Map<?, ? extends Number> m1, @NonNull Map<?, ? extends Number> m2) {
         return 1.0 - Math.acos(Cosine.calculate(m1, m2)) / Math.PI;
      }

      @Override
      public DistanceMeasure asDistanceMeasure() {
         return Distance.Angular;
      }
   }


}//END OF SimilarityMeasures
