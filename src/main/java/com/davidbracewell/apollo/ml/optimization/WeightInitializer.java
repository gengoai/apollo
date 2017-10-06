package com.davidbracewell.apollo.ml.optimization;


import com.davidbracewell.apollo.linear.NDArray;

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
      double max = Math.sqrt(6.0) / Math.sqrt(m.shape().i + m.shape().j);
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
   NDArray initialize(NDArray weights);


}// END OF WeightInitializer
