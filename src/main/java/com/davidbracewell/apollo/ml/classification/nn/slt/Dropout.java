package com.davidbracewell.apollo.ml.classification.nn.slt;

import com.davidbracewell.apollo.linalg.DenseFloatMatrix;
import com.davidbracewell.apollo.linalg.Matrix;
import lombok.Getter;
import lombok.val;

/**
 * @author David B. Bracewell
 */
public class Dropout extends Layer {
   double rate;

   protected Dropout(int inputSize, int outputSize, double rate) {
      super(inputSize, outputSize);
      this.rate = rate;
   }

   public static DropoutBuilder builder() {
      return new DropoutBuilder();
   }

   public static DropoutBuilder builder(double rate) {
      return new DropoutBuilder().rate(rate);
   }

   @Override
   Matrix backward(Matrix input, Matrix output, Matrix delta, double learningRate, int layerIndex) {
      return delta;
   }

   @Override
   Matrix forward(Matrix input) {
      val mask = DenseFloatMatrix.rand(input.toFloatArray().length).predicate(x -> x > rate);
      return input.muli(mask);
   }

   @Override
   public boolean trainOnly() {
      return true;
   }

   public static class DropoutBuilder extends LayerBuilder<DropoutBuilder> {
      @Getter
      double rate = 0.5;

      @Override
      public Layer build() {
         return new Dropout(getInputSize(), getOutputSize(), getRate());
      }

      public DropoutBuilder rate(double rate) {
         this.rate = rate;
         return this;
      }
   }

}// END OF Dropout
