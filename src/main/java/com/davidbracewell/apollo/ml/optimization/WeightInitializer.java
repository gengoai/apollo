package com.davidbracewell.apollo.ml.optimization;


import com.davidbracewell.apollo.linear.NDArray;
import lombok.NonNull;

import java.io.Serializable;
import java.util.Random;

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
      double max = Math.sqrt(6.0) / Math.sqrt(m.numRows() + m.numCols());
      double min = -max;
      m.mapi(x -> min + (max - min) * Math.random());
      return m;
   };

   WeightInitializer ZEROES = (m) -> m.mapi(x -> 0d);


   static WeightInitializer RAND(@NonNull Random rnd) {
      return (m) -> m.map(d -> rnd.nextDouble());
   }

   static WeightInitializer RAND() {
      return (m) -> m.map(d -> Math.random());
   }

   static WeightInitializer RAND(@NonNull Random rnd, int min, int max) {
      return (m) -> m.map(d -> min + rnd.nextDouble() * max);
   }

   static WeightInitializer RAND(int min, int max) {
      return RAND(new Random(), min, max);
   }

   static WeightInitializer RANDN(@NonNull Random rnd) {
      return (m) -> m.map(d -> rnd.nextGaussian());
   }

   static WeightInitializer RANDN() {
      return RANDN(new Random());
   }

   /**
    * Initialize.
    *
    * @param weights the weights
    */
   NDArray initialize(NDArray weights);


}// END OF WeightInitializer
