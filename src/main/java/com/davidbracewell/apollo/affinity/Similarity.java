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

import com.davidbracewell.apollo.stats.ContingencyTable;
import com.google.common.base.Preconditions;
import lombok.NonNull;

import java.util.Map;

/**
 * The enum Similarity.
 *
 * @author David B. Bracewell
 */
public enum Similarity implements SimilarityMeasure {
  /**
   * The Dot product.
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
   * The Dice.
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
      Preconditions.checkArgument(table.rowCount() == table.columnCount() && table.rowCount() == 2, "Only supports 2x2 contingency tables.");
      return 2 * table.get(0, 0) / (table.columnSum(0) + table.rowSum(0));
    }
  },
  /**
   * The Cosine.
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
      Preconditions.checkArgument(table.rowCount() == table.columnCount() && table.rowCount() == 2, "Only supports 2x2 contingency tables.");
      return table.get(0, 0) / Math.sqrt(table.columnSum(0) + table.rowSum(0));
    }
  },
  /**
   * The Jaccard.
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
      Preconditions.checkArgument(table.rowCount() == table.columnCount() && table.rowCount() == 2, "Only supports 2x2 contingency tables.");
      return table.get(0, 0) / (table.get(0, 0) + table.get(0, 1) + table.get(1, 0));
    }
  },
  /**
   * The Overlap.
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

      return dp / Math.min(mag1, mag2);
    }

    @Override
    public double calculate(@NonNull ContingencyTable table) {
      Preconditions.checkNotNull(table);
      Preconditions.checkArgument(table.rowCount() == table.columnCount() && table.rowCount() == 2, "Only supports 2x2 contingency tables.");
      return table.get(0, 0) / Math.min(table.columnSum(0), table.rowSum(0));
    }
  },
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
