package com.davidbracewell.apollo.ml.nn;

import com.davidbracewell.apollo.linalg.Vector;
import com.davidbracewell.apollo.optimization.WeightInitializer;
import com.davidbracewell.apollo.optimization.Weights;
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
   private final Weights weights;

   public WeightLayer(int inputSize, int outputSize, Activation activation, WeightInitializer weightInitializer) {
      super(inputSize, outputSize);
      this.activation = activation;
      this.weights = new Weights(outputSize, inputSize, weightInitializer);
   }

   @Override
   public Vector backward(Vector output, Vector delta) {
      return delta.multiplySelf(activation.valueGradient(output))
                  .toMatrix()
                  .multiply(weights.getTheta())
                  .row(0);
   }

   @Override
   public Vector forward(Vector input) {
      return activation.apply(weights.dot(input));
   }

   @Override
   public Weights getWeights() {
      return weights;
   }

   @Override
   public void setWeights(Weights weights) {
      this.weights.set(weights);
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
