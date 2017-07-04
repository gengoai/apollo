package com.davidbracewell.apollo.optimization.o2;

import com.davidbracewell.apollo.linalg.Matrix;
import com.davidbracewell.apollo.optimization.Weights;
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

   /**
    * The constant DEFAULT.
    */
   WeightInitializer DEFAULT = (m) -> {
      double max = 1d / (m.numberOfRows() + m.numberOfColumns());
      double min = -max;
      for (int r = 0; r < m.numberOfRows(); r++) {
         for (int c = 0; c < m.numberOfColumns(); c++) {
            m.set(r, c, min + (max - min) * Math.random());
         }
      }
   };

}// END OF WeightInitializer
