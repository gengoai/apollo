package com.gengoai.apollo.linear.v2;

import com.gengoai.function.SerializableConsumer;
import lombok.NonNull;

import java.util.Random;

/**
 * @author David B. Bracewell
 */
public interface NDArrayInitializer extends SerializableConsumer<NDArray> {

   /**
    * Glorot and Bengio (2010) for sigmoid units
    */
   NDArrayInitializer glorotAndBengioSigmoid = (m) -> {
      double max = 4 * Math.sqrt(6.0) / Math.sqrt(m.numRows() + m.numCols());
      double min = -max;
      m.mapi(x -> min + (max - min) * Math.random());
   };

   /**
    * Glorot and Bengio (2010) for hyperbolic tangent units
    */
   NDArrayInitializer glorotAndBengioTanH = (m) -> {
      double max = Math.sqrt(6.0) / Math.sqrt(m.numRows() + m.numCols());
      double min = -max;
      m.mapi(x -> min + (max - min) * Math.random());
   };

   /**
    * Rand nd array initializer.
    *
    * @param rnd the rnd
    * @return the nd array initializer
    */
   static NDArrayInitializer rand(@NonNull Random rnd) {
      return (m) -> m.mapi(d -> rnd.nextDouble());
   }

   /**
    * Rand nd array initializer.
    */
   NDArrayInitializer rand = (m) -> m.mapi(d -> Math.random());

   /**
    * Rand nd array initializer.
    *
    * @param rnd the rnd
    * @param min the min
    * @param max the max
    * @return the nd array initializer
    */
   static NDArrayInitializer rand(@NonNull Random rnd, int min, int max) {
      return (m) -> m.mapi(d -> min + rnd.nextDouble() * max);
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
   static NDArrayInitializer randn(Random rnd) {
      return (m) -> m.mapi(d -> rnd.nextGaussian());
   }

   /**
    * Randn nd array initializer.
    */
   NDArrayInitializer randn = randn(new Random());

   /**
    * The constant ZEROES.
    */
   NDArrayInitializer zeroes = (m) -> m.mapi(x -> 0d);

   /**
    * The constant Ones.
    */
   NDArrayInitializer ones = (m) -> m.mapi(x -> 1d);


}//END OF NDArrayInitializer
