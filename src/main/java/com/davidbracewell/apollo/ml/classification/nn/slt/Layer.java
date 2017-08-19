package com.davidbracewell.apollo.ml.classification.nn.slt;

import com.davidbracewell.apollo.linalg.Matrix;
import com.davidbracewell.conversion.Cast;
import lombok.Getter;

import java.io.Serializable;

/**
 * @author David B. Bracewell
 */
public abstract class Layer implements Serializable {
   private final int inputSize;
   private final int outputSize;

   protected Layer(int inputSize, int outputSize) {
      this.inputSize = inputSize;
      this.outputSize = outputSize;
   }

   /**
    * Backward vector.
    *
    * @param output the output
    * @param delta  the delta
    * @return the vector
    */
   abstract Matrix backward(Matrix input, Matrix output, Matrix delta, double learningRate, int layerIndex);

   /**
    * Forward vector.
    *
    * @param input the input
    * @return the vector
    */
   abstract Matrix forward(Matrix input);

   public boolean trainOnly() {
      return false;
   }

   protected static abstract class LayerBuilder<T extends LayerBuilder> implements Serializable {
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
