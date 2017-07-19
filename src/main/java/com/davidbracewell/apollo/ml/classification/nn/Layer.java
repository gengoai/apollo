package com.davidbracewell.apollo.ml.classification.nn;

import com.davidbracewell.apollo.linalg.Vector;
import com.davidbracewell.apollo.optimization.WeightMatrix;
import com.davidbracewell.conversion.Cast;
import lombok.AllArgsConstructor;
import lombok.Getter;

import java.io.Serializable;

/**
 * @author David B. Bracewell
 */
@AllArgsConstructor
public abstract class Layer implements Serializable {
   private static final long serialVersionUID = 1L;
   @Getter
   private final int inputSize;
   @Getter
   private final int outputSize;


   /**
    * Backward vector.
    *
    * @param output the output
    * @param delta  the delta
    * @return the vector
    */
   abstract Vector backward(Vector output, Vector delta);

   /**
    * Forward vector.
    *
    * @param input the input
    * @return the vector
    */
   abstract Vector forward(Vector input);

   /**
    * Gets weights.
    *
    * @return the weights
    */
   public WeightMatrix getWeights() {
      return null;
   }

   /**
    * Has weights boolean.
    *
    * @return the boolean
    */
   public boolean hasWeights() {
      return false;
   }

   public static abstract class LayerBuilder<T extends LayerBuilder> implements Serializable {
      private static final long serialVersionUID = 1L;
      @Getter
      private int inputSize;
      @Getter
      private int outputSize;

      public abstract Layer build();


      public T inputSize(int inputSize) {
         this.inputSize = inputSize;
         return Cast.as(this);
      }

      public T outputSize(int outputSize) {
         this.outputSize = outputSize;
         return Cast.as(this);
      }
   }

}// END OF Layer
