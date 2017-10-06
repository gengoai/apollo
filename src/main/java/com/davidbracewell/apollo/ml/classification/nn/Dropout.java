package com.davidbracewell.apollo.ml.classification.nn;

import com.davidbracewell.apollo.linear.NDArray;
import com.davidbracewell.apollo.linear.NDArrayFactory;
import com.davidbracewell.apollo.ml.optimization.WeightUpdate;
import com.davidbracewell.tuple.Tuple2;
import lombok.Getter;
import lombok.val;

import static com.davidbracewell.tuple.Tuples.$;

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
   public NDArray backward(NDArray input, NDArray output, NDArray delta, double learningRate, int layerIndex, int iteration) {
      return delta;
   }

   @Override
   public BackpropResult backward(NDArray input, NDArray output, NDArray delta, boolean calculateDelta) {
      return BackpropResult.from(delta, NDArrayFactory.defaultFactory().empty(),
                                 NDArrayFactory.defaultFactory().empty());
   }

   @Override
   public Tuple2<NDArray, Double> backward(WeightUpdate updater, NDArray input, NDArray output, NDArray delta, int iteration, boolean calcuateDelta) {
      return $(delta, 0d);
   }

   @Override
   public Layer copy() {
      return new Dropout(getInputSize(), getOutputSize(), rate);
   }

   @Override
   NDArray forward(NDArray input) {
      val mask = NDArrayFactory.defaultFactory().rand(input.numRows(), input.numCols())
                               .test(x -> x < rate);
      return input.mul(mask).divi(rate);
   }

   @Override
   public NDArray getBias() {
      return NDArrayFactory.defaultFactory().empty();
   }

   @Override
   public NDArray getWeights() {
      return NDArrayFactory.defaultFactory().empty();
   }

   @Override
   public boolean trainOnly() {
      return true;
   }

   @Override
   public double update(WeightUpdate weightUpdate, NDArray wGrad, NDArray bBrad, int iteration) {
      return 0;
   }

   @Override
   public void update(NDArray[] weights, NDArray[] bias) {

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
