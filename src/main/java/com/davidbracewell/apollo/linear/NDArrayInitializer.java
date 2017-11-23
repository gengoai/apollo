package com.davidbracewell.apollo.linear;


import lombok.NonNull;

import java.io.Serializable;
import java.util.Random;

/**
 * The interface Weight initializer.
 *
 * @author David B. Bracewell
 */
@FunctionalInterface
public interface NDArrayInitializer extends Serializable {

   /**
    * Glorot and Bengio (2010) for sigmoid units
    */
   static NDArrayInitializer glorotAndBengioSigmoid() {
      return (m) -> {
         double max = 4 * Math.sqrt(6.0) / Math.sqrt(m.numRows() + m.numCols());
         double min = -max;
         m.mapi(x -> min + (max - min) * Math.random());
         return m;
      };
   }

   /**
    * Glorot and Bengio (2010) for hyperbolic tangent units
    */
   static NDArrayInitializer glorotAndBengioTanH() {
      return (m) -> {
         double max = Math.sqrt(6.0) / Math.sqrt(m.numRows() + m.numCols());
         double min = -max;
         m.mapi(x -> min + (max - min) * Math.random());
         return m;
      };
   }

   /**
    * Rand nd array initializer.
    *
    * @param rnd the rnd
    * @return the nd array initializer
    */
   static NDArrayInitializer rand(@NonNull Random rnd) {
      return (m) -> m.map(d -> rnd.nextDouble());
   }

   /**
    * Rand nd array initializer.
    *
    * @return the nd array initializer
    */
   static NDArrayInitializer rand() {
      return (m) -> m.map(d -> Math.random());
   }

   /**
    * Rand nd array initializer.
    *
    * @param rnd the rnd
    * @param min the min
    * @param max the max
    * @return the nd array initializer
    */
   static NDArrayInitializer rand(@NonNull Random rnd, int min, int max) {
      return (m) -> m.map(d -> min + rnd.nextDouble() * max);
   }

   /**
    * Rand nd array initializer.
    *
    * @param min the min
    * @param max the max
    * @return the nd array initializer
    */
   static NDArrayInitializer rand(int min, int max) {
      return rand(new Random(), min, max);
   }

   /**
    * Randn nd array initializer.
    *
    * @param rnd the rnd
    * @return the nd array initializer
    */
   static NDArrayInitializer randn(@NonNull Random rnd) {
      return (m) -> m.map(d -> rnd.nextGaussian());
   }

   /**
    * Randn nd array initializer.
    *
    * @return the nd array initializer
    */
   static NDArrayInitializer randn() {
      return randn(new Random());
   }

   /**
    * The constant ZEROES.
    */
   static NDArrayInitializer zeroes() {
      return (m) -> m.mapi(x -> 0d);
   }

   /**
    * Initialize.
    *
    * @param weights the weights
    * @return the nd array
    */
   NDArray initialize(NDArray weights);


}// END OF WeightInitializer
