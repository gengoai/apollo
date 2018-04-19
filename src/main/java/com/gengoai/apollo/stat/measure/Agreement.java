package com.gengoai.apollo.stat.measure;

import com.gengoai.guava.common.base.Preconditions;
import lombok.NonNull;

/**
 * @author David B. Bracewell
 */
public enum Agreement implements ContingencyTableCalculator {
   Cohen_Kappa {
      @Override
      public double calculate(@NonNull ContingencyTable table) {
         Preconditions.checkArgument(table.columnCount() == 2
                                        && table.rowCount() == 2,
                                     "Only 2x2 tables supported");
         double sum = table.getSum();
         double sumSq = sum * sum;
         double Po = (table.get(0, 0) + table.get(1, 1)) / sum;
         double Pe = ((table.columnSum(0) * table.rowSum(0)) / sumSq)
                        + ((table.columnSum(1) * table.rowSum(1)) / sumSq);
         return (Po - Pe) / (1.0 - Pe);
      }
   };


}// END OF Agreement
