package com.davidbracewell.apollo.ml.classification.nn;

import com.davidbracewell.apollo.linalg.Vector;
import com.davidbracewell.apollo.optimization.GradientMatrix;
import com.davidbracewell.apollo.optimization.WeightInitializer;
import com.davidbracewell.apollo.optimization.WeightMatrix;
import com.davidbracewell.apollo.optimization.activation.Activation;
import com.davidbracewell.conversion.Cast;
import lombok.Getter;
import lombok.NonNull;

/**
 * @author David B. Bracewell
 */
public abstract class WeightLayer extends Layer {
   private static final long serialVersionUID = 1L;
   private final Activation activation;
   private final WeightMatrix weights;

   public WeightLayer(int inputSize, int outputSize, Activation activation, WeightInitializer weightInitializer) {
      super(inputSize, outputSize);
      this.activation = activation;
      this.weights = weightInitializer.initialize(new WeightMatrix(outputSize, inputSize));
   }

   @Override
   public Vector backward(Vector input, Vector output, Vector delta) {
      delta.multiplySelf(activation.valueGradient(output));
      if (getGradient() == null) {
         setGradient(GradientMatrix.calculate(input, delta));
      } else {
         getGradient().add(GradientMatrix.calculate(input, delta));
      }
      return weights.backward(delta);
   }

   @Override
   public Vector forward(Vector input) {
      return weights.dot(input, activation);
   }

   @Override
   public WeightMatrix getWeights() {
      return weights;
   }

   @Override
   public boolean hasWeights() {
      return true;
   }

   protected static abstract class WeightLayerBuilder<T extends WeightLayerBuilder> extends LayerBuilder<T> {
      @Getter
      private Activation activation = Activation.SIGMOID;
      @Getter
      private WeightInitializer weightInitializer = WeightInitializer.DEFAULT;

      public T activation(@NonNull Activation activation) {
         this.activation = activation;
         return Cast.as(this);
      }

      public T weightInitializer(@NonNull WeightInitializer weightInitializer) {
         this.weightInitializer = weightInitializer;
         return Cast.as(this);
      }

   }

}// END OF WeightLayer
