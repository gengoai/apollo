package com.davidbracewell.apollo.ml.nn;

import com.davidbracewell.apollo.linalg.Vector;
import lombok.Getter;
import lombok.Setter;
import lombok.experimental.Accessors;

/**
 * @author David B. Bracewell
 */
public class Dropout extends TrainOnlyLayer {
   private static final long serialVersionUID = 1L;
   private final double dropoutRate;

   public Dropout(int inputSize, int outputSize, double dropoutRate) {
      super(inputSize, outputSize);
      this.dropoutRate = dropoutRate;
   }

   public static DropoutBuilder builder() {
      return new DropoutBuilder();
   }

   @Override
   Vector backward(Vector output, Vector delta) {
      return delta;
   }

   @Override
   Vector forward(Vector input) {
      Vector v = Vector.sZeros(input.dimension());
      for (int i = 0; i < v.dimension(); i++) {
         v.set(i, Math.random() >= dropoutRate ? 1.0 : 0.0);
      }
      return input.multiply(v);
   }

   @Accessors(fluent = true)
   public static class DropoutBuilder extends Layer.LayerBuilder<DropoutBuilder> {
      @Getter
      @Setter
      private double dropoutRate = 0.2;

      @Override
      public Layer build() {
         return new Dropout(getInputSize(), getOutputSize(), dropoutRate);
      }
   }

}// END OF Dropout
