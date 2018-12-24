package com.gengoai.apollo.ml.neural;

import com.gengoai.Copyable;
import com.gengoai.apollo.linear.NDArray;
import com.gengoai.apollo.ml.data.Dataset;
import com.gengoai.apollo.optimization.WeightUpdate;
import com.gengoai.conversion.Cast;
import com.gengoai.tuple.Tuple2;

import java.io.Serializable;

/**
 * @author David B. Bracewell
 */
public abstract class Layer implements Serializable, Copyable<Layer> {
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
   public abstract NDArray backward(NDArray input, NDArray output, NDArray delta, double learningRate, int layerIndex, int iteration);

   public abstract Tuple2<NDArray, Double> backward(WeightUpdate updater, NDArray input, NDArray output, NDArray delta, int iteration, boolean calcuateDelta);

   public abstract BackpropResult backward(NDArray input, NDArray output, NDArray delta, boolean calculateDelta);

   /**
    * Forward vector.
    *
    * @param input the input
    * @return the vector
    */
   abstract NDArray forward(NDArray input);

   public abstract NDArray getBias();

   public int getInputSize() {
      return inputSize;
   }

   public int getOutputSize() {
      return outputSize;
   }

   public abstract NDArray getWeights();

   public void preTrain(Dataset dataset) {

   }

   public boolean trainOnly() {
      return false;
   }

   public abstract double update(WeightUpdate weightUpdate, NDArray wGrad, NDArray bBrad, int iteration);

   public abstract void update(NDArray[] weights, NDArray[] bias);

   protected static abstract class LayerBuilder<T extends LayerBuilder> implements Serializable {
      private static final long serialVersionUID = 1L;
      private int inputSize;
      private int outputSize;

      public abstract Layer build();

      public int getInputSize() {
         return inputSize;
      }

      public int getOutputSize() {
         return outputSize;
      }

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
