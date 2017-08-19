package com.davidbracewell.apollo.optimization;

import com.davidbracewell.apollo.linalg.Matrix;

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
      double max = Math.sqrt(6.0) / Math.sqrt(m.numCols() + m.numRows());
      double min = -max;
      m.mapi(x -> min + (max - min) * Math.random());
      return m;
   };

   WeightInitializer ZEROES = (m) -> m.mapi(x -> 0d);

   /**
    * Initialize.
    *
    * @param weights the weights
    */
   Matrix initialize(Matrix weights);


}// END OF WeightInitializer
