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

import com.davidbracewell.apollo.linalg.DenseVector;
import com.davidbracewell.apollo.linalg.Vector;
import com.davidbracewell.collection.EnhancedDoubleStatistics;
import com.google.common.base.Preconditions;
import lombok.NonNull;

import java.util.ArrayList;
import java.util.Collections;

/**
 * @author David B. Bracewell
 */
public enum Correlation implements CorrelationMeasure {
  Pearson {
    @Override
    public double calculate(@NonNull double[] v1, @NonNull double[] v2) {
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
  },
  Spearman {
    @Override
    public double calculate(@NonNull double[] v1, @NonNull double[] v2) {
      Preconditions.checkArgument(v1.length == v2.length, "Vector dimension mismatch " + v1.length + " != " + v2.length);
      Preconditions.checkArgument(v1.length >= 2, "Need at least two elements");
      return Pearson.calculate(rank(v1), rank(v2));
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
      } else if (i < v.length-1 && rp.value == rankPairs.get(i + 1).value) {
        rank = (rank + i + 2) / 2d;
      }
      out[rp.index] = rank;
    }
    return out;
  }

  private static class RankPair implements Comparable<RankPair> {
    final int index;
    final double value;

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
