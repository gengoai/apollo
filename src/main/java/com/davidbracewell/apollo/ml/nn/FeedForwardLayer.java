package com.davidbracewell.apollo.ml.nn;

import com.davidbracewell.apollo.linalg.Matrix;
import com.davidbracewell.apollo.optimization.activation.DifferentiableActivation;
import lombok.NonNull;

/**
 * @author David B. Bracewell
 */
public class FeedForwardLayer implements Layer {
   private final DifferentiableActivation activation;
   private Matrix forwardResult = null;

   public FeedForwardLayer(@NonNull DifferentiableActivation activation) {
      this.activation = activation;
   }

   @Override
   public Matrix backward(Matrix m) {
      return activation.valueGradient(forwardResult, m);
   }

   @Override
   public Matrix forward(Matrix m) {
      this.forwardResult = m.map(activation::apply);
      return this.forwardResult;
   }

}// END OF FeedForwardLayer
