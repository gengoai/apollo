package com.davidbracewell.apollo.optimization.o2;

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
    * Initialize.
    *
    * @param weights the weights
    */
   void initialize(Matrix weights);

}// END OF WeightInitializer
