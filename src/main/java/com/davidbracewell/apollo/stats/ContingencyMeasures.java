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

package com.davidbracewell.apollo.stats;

import com.google.common.base.Preconditions;
import com.google.common.math.DoubleMath;
import com.google.common.primitives.Doubles;

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
    public double calculate(ContingencyTable table) {
      Preconditions.checkNotNull(table);
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
    public double calculate(ContingencyTable table) {
      Preconditions.checkNotNull(table);
      Preconditions.checkArgument(table.rowCount() == table.columnCount() && table.rowCount() == 2, "Only supports 2x2 contingency tables.");
      return DoubleMath.log2(table.get(0, 0)) - DoubleMath.log2(table.getExpected(0, 0));
    }
  },
  /**
   * The ODDS_RATIO.
   */
  ODDS_RATIO {
    @Override
    public double calculate(ContingencyTable table) {
      Preconditions.checkNotNull(table);
      Preconditions.checkArgument(table.rowCount() == table.columnCount() && table.rowCount() == 2, "Only supports 2x2 contingency tables.");
      double n21 = table.get(1, 0);
      if (n21 == 0) {
        n21 = 1;
      }
      double n12 = table.get(0, 1);
      if (n12 == 0) {
        n12 = 1;
      }
      return (table.get(0, 0) * table.get(1, 1)) / (n21 * n12);
    }
  },
  /**
   * The T_SCORE.
   */
  T_SCORE {
    @Override
    public double calculate(ContingencyTable table) {
      Preconditions.checkNotNull(table);
      Preconditions.checkArgument(table.rowCount() == table.columnCount() && table.rowCount() == 2, "Only supports 2x2 contingency tables.");
      return (table.get(0, 0) - table.getExpected(0, 0)) / Math.sqrt(table.get(0, 0));
    }
  },
  /**
   * The NPMI.
   */
  NPMI {
    @Override
    public double calculate(ContingencyTable table) {
      Preconditions.checkNotNull(table);
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
    public double calculate(ContingencyTable table) {
      Preconditions.checkNotNull(table);
      Preconditions.checkArgument(table.rowCount() == table.columnCount() && table.rowCount() == 2, "Only supports 2x2 contingency tables.");
      return table.get(0, 0) * (Math.log(table.get(0, 0) / table.getExpected(0, 0)) - 1);
    }
  },
  /**
   * The CHI_SQUARE.
   */
  CHI_SQUARE {
    @Override
    public double calculate(ContingencyTable table) {
      Preconditions.checkNotNull(table);
      double sumSq = 0d;
      for (int row = 0; row < table.rowCount(); row++) {
        for (int col = 0; col < table.columnCount(); col++) {
          double expected = table.getExpected(row, col);
          sumSq += Math.pow(table.get(row, col) - expected, 2) / expected;
        }
      }
      return Doubles.isFinite(sumSq) ? sumSq : 0d;
    }
  },
  /**
   * The LOG_LIKELIHOOD.
   */
  LOG_LIKELIHOOD {
    @Override
    public double calculate(ContingencyTable table) {
      Preconditions.checkNotNull(table);
      double sum = 0d;
      for (int row = 0; row < table.rowCount(); row++) {
        for (int col = 0; col < table.columnCount(); col++) {
          sum += table.get(row, col) * Math.log(table.get(row, col) / table.getExpected(row, col));
        }
      }
      return Doubles.isFinite(sum) ? 2 * sum : 0d;
    }
  },
  /**
   * The RELATIVE_RISK.
   */
  RELATIVE_RISK {
    @Override
    public double calculate(ContingencyTable table) {
      return (table.get(0, 0) / (table.get(0, 0) + table.get(0, 1))) / (table.get(1, 0) / (table.get(1, 0) + table.get(1, 1)));
    }
  }


}//END OF AssociationMeasures
