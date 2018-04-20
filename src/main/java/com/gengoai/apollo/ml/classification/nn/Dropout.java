package com.gengoai.apollo.ml.classification.nn;

import com.gengoai.apollo.linear.NDArray;
import com.gengoai.apollo.linear.NDArrayFactory;
import com.gengoai.apollo.linear.NDArrayInitializer;
import com.gengoai.apollo.ml.optimization.WeightUpdate;
import com.gengoai.tuple.Tuple2;
import com.gengoai.apollo.ml.optimization.WeightUpdate;
import lombok.Getter;
import lombok.val;

import static com.gengoai.tuple.Tuples.$;

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
      return BackpropResult.from(delta, NDArrayFactory.DEFAULT().empty(),
                                 NDArrayFactory.DEFAULT().empty());
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
      val mask = NDArrayFactory.DEFAULT().create(input.numRows(), input.numCols(), NDArrayInitializer.rand())
                               .test(x -> x < rate);
      return input.mul(mask).divi(rate);
   }

   @Override
   public NDArray getBias() {
      return NDArrayFactory.DEFAULT().empty();
   }

   @Override
   public NDArray getWeights() {
      return NDArrayFactory.DEFAULT().empty();
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
