package com.davidbracewell.apollo.ml.classification.nn;

import com.davidbracewell.apollo.linalg.Vector;
import com.davidbracewell.apollo.optimization.GradientMatrix;
import com.davidbracewell.apollo.optimization.WeightInitializer;
import com.davidbracewell.apollo.optimization.activation.Activation;
import com.davidbracewell.apollo.optimization.activation.SigmoidActivation;
import com.davidbracewell.apollo.optimization.activation.SoftmaxActivation;
import lombok.NonNull;

/**
 * @author David B. Bracewell
 */
public class OutputLayer extends WeightLayer {
   private static final long serialVersionUID = 1L;


   public OutputLayer(int inputSize, int outputSize, @NonNull Activation activation, @NonNull WeightInitializer weightInitializer) {
      super(inputSize, outputSize, activation, weightInitializer);
   }

   public static Builder builder() {
      return new Builder();
   }

   public static Builder sigmoid() {
      return new Builder().activation(new SigmoidActivation());
   }

   public static Builder softmax() {
      return new Builder().activation(new SoftmaxActivation());
   }

   @Override
   public Vector backward(Vector input, Vector output, Vector delta) {
      Vector deltaOut = getWeights().backward(delta);
      if (getGradient() == null) {
         setGradient(GradientMatrix.calculate(input, delta));
      } else {
         getGradient().add(GradientMatrix.calculate(input, delta));
      }
      return deltaOut;
   }

   public static class Builder extends WeightLayerBuilder<Builder> {

      @Override
      public Layer build() {
         return new OutputLayer(getInputSize(), getOutputSize(), getActivation(), getWeightInitializer());
      }
   }

}// END OF OutputLayer
