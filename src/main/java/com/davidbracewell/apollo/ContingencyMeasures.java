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

package com.davidbracewell.apollo;

import com.davidbracewell.apollo.distribution.NormalDistribution;
import com.google.common.base.Preconditions;
import com.google.common.math.DoubleMath;
import com.google.common.primitives.Doubles;
import lombok.NonNull;
import org.apache.commons.math.MathException;
import org.apache.commons.math.distribution.ChiSquaredDistribution;
import org.apache.commons.math.distribution.ChiSquaredDistributionImpl;
import org.apache.commons.math.distribution.TDistribution;
import org.apache.commons.math.distribution.TDistributionImpl;

/**
 * The enum Contingency measures.
 *
 * @author David B. Bracewell
 */
public enum ContingencyMeasures implements ContingencyTableCalculator {
  /**
   * The MI.
   */
  MI {
    @Override
    public double calculate(@NonNull ContingencyTable table) {
      double sum = 0d;
      for (int row = 0; row < table.rowCount(); row++) {
        for (int col = 0; col < table.columnCount(); col++) {
          sum += table.get(row, col) / table.getSum() * DoubleMath.log2(table.get(row, col) / table.getExpected(row, col));
        }
      }
      return Doubles.isFinite(sum) ? sum : 0d;
    }
  },
  /**
   * The PMI.
   */
  PMI {
    @Override
    public double calculate(@NonNull ContingencyTable table) {
      Preconditions.checkArgument(table.rowCount() == table.columnCount() && table.rowCount() == 2, "Only supports 2x2 contingency tables.");
      return DoubleMath.log2(table.get(0, 0)) - DoubleMath.log2(table.getExpected(0, 0));
    }
  },
  /**
   * The ODDS_RATIO.
   */
  ODDS_RATIO {
    @Override
    public double calculate(@NonNull ContingencyTable table) {
      Preconditions.checkArgument(table.rowCount() == table.columnCount() && table.rowCount() == 2, "Only supports 2x2 contingency tables.");
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
   * The T_SCORE.
   */
  T_SCORE {
    @Override
    public double calculate(@NonNull ContingencyTable table) {
      Preconditions.checkArgument(table.rowCount() == table.columnCount() && table.rowCount() == 2, "Only supports 2x2 contingency tables.");
      return (table.get(0, 0) - table.getExpected(0, 0)) / Math.sqrt(table.get(0, 0));
    }

    @Override
    public double pValue(@NonNull ContingencyTable table) {
      TDistribution distribution = new TDistributionImpl(table.degreesOfFreedom());
      try {
        return 1.0 - distribution.cumulativeProbability(calculate(table));
      } catch (MathException e) {
        return Double.POSITIVE_INFINITY;
      }
    }

  },
  /**
   * The NPMI.
   */
  NPMI {
    @Override
    public double calculate(@NonNull ContingencyTable table) {
      Preconditions.checkArgument(table.rowCount() == table.columnCount() && table.rowCount() == 2, "Only supports 2x2 contingency tables.");
      if (table.get(0, 0) == 0) {
        return -1;
      }
      return DoubleMath.log2(table.get(0, 0) / table.getExpected(0, 0)) /
        -DoubleMath.log2(table.get(0, 0) / table.getSum());
    }
  },
  /**
   * The POISSON_STIRLING.
   */
  POISSON_STIRLING {
    @Override
    public double calculate(@NonNull ContingencyTable table) {
      Preconditions.checkArgument(table.rowCount() == table.columnCount() && table.rowCount() == 2, "Only supports 2x2 contingency tables.");
      return table.get(0, 0) * (Math.log(table.get(0, 0) / table.getExpected(0, 0)) - 1);
    }
  },
  /**
   * The CHI_SQUARE.
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
      ChiSquaredDistribution distribution = new ChiSquaredDistributionImpl(table.degreesOfFreedom());
      try {
        return 1.0 - distribution.cumulativeProbability(calculate(table));
      } catch (MathException e) {
        return Double.POSITIVE_INFINITY;
      }
    }
  },
  /**
   * The G_SQUARE or likelihood test.
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
      ChiSquaredDistribution distribution = new ChiSquaredDistributionImpl(table.degreesOfFreedom());
      try {
        return 1.0 - distribution.cumulativeProbability(calculate(table));
      } catch (MathException e) {
        return Double.POSITIVE_INFINITY;
      }
    }
  },
  /**
   * The RELATIVE_RISK.
   */
  RELATIVE_RISK {
    @Override
    public double calculate(@NonNull ContingencyTable table) {
      Preconditions.checkArgument(table.rowCount() == table.columnCount() && table.rowCount() == 2, "Only supports 2x2 contingency tables.");
      double v1 = table.get(0, 0) / table.rowSum(0);
      double v2 = table.get(1, 0) / table.rowSum(1);
      return v1 / v2;
    }

    @Override
    public double pValue(@NonNull ContingencyTable table) {
      NormalDistribution distribution = new NormalDistribution(0, 1);
      return 1.0 - distribution.cumulativeProbability(Math.log(calculate(table)));
    }
  };


  public static void main(String[] args) {
    ContingencyTable table = new ContingencyTable(2, 2);
    table.set(0, 0, 139);
    table.set(0, 1, 10898);
    table.set(1, 0, 239);
    table.set(1, 1, 10795);

    double rr = RELATIVE_RISK.calculate(table);
    System.out.println(rr);
    System.out.println(Math.log(rr));
    System.out.println(RELATIVE_RISK.pValue(table));

  }

}//END OF AssociationMeasures
