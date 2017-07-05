package com.davidbracewell.apollo.optimization;

import com.davidbracewell.apollo.linalg.Matrix;
import lombok.NonNull;

import java.io.Serializable;

/**
 * The interface Weight initializer.
 *
 * @author David B. Bracewell
 */
@FunctionalInterface
public interface WeightInitializer extends Serializable {

   /**
    * The constant DEFAULT.
    */
   WeightInitializer DEFAULT = (m) -> {
      double max = 1.0 / Math.sqrt(6.0 / m.numberOfColumns() + m.numberOfRows());
      double min = -max;
      for (int r = 0; r < m.numberOfRows(); r++) {
         for (int c = 0; c < m.numberOfColumns(); c++) {
            m.set(r, c, min + (max - min) * Math.random());
         }
      }
   };

   WeightInitializer ZEROES = (m) -> m.nonZeroIterator().forEachRemaining(e -> m.set(e.row, e.column, 0));

   /**
    * Initialize.
    *
    * @param weights the weights
    */
   void initialize(Matrix weights);

   /**
    * Initialize.
    *
    * @param weights the weights
    */
   default void initialize(@NonNull Weights weights) {
      initialize(weights.getTheta());
   }

}// END OF WeightInitializer
