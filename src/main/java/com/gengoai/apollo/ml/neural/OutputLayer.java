package com.gengoai.apollo.ml.neural;

import com.gengoai.apollo.linear.Axis;
import com.gengoai.apollo.linear.NDArray;
import com.gengoai.apollo.linear.NDArrayInitializer;
import com.gengoai.apollo.optimization.WeightUpdate;
import com.gengoai.apollo.optimization.activation.Activation;
import com.gengoai.apollo.optimization.activation.SigmoidActivation;
import com.gengoai.apollo.optimization.activation.SoftmaxActivation;
import com.gengoai.tuple.Tuple2;

/**
 * @author David B. Bracewell
 */
public class OutputLayer extends WeightLayer {
   public OutputLayer(int inputSize, int outputSize, Activation activation, NDArrayInitializer NDArrayInitializer, double l1, double l2) {
      super(inputSize, outputSize, activation, NDArrayInitializer, l1, l2);
   }

   public OutputLayer(WeightLayer layer) {
      super(layer);
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
   public Tuple2<NDArray, Double> backward(WeightUpdate updater, NDArray input, NDArray output, NDArray delta, int iteration, boolean calcuateDelta) {
      return updater.update(this, input, output, delta, iteration, calcuateDelta);
   }

   @Override
   public BackpropResult backward(NDArray input, NDArray output, NDArray delta, boolean calculateDelta) {
      NDArray dzOut = calculateDelta
                      ? weights.T().mmul(delta)
                      : null;
      NDArray dw = delta.mmul(input.T());
      NDArray db = delta.sum(Axis.ROW);
      return BackpropResult.from(dzOut, dw, db);
   }

   @Override
   public NDArray backward(NDArray input, NDArray output, NDArray delta, double learningRate, int layerIndex, int iteration) {
      NDArray dzOut = layerIndex > 0
                      ? weights.T().mmul(delta)
                      : null;
      NDArray dw = delta.mmul(input.T())
                        .divi(input.numCols());
      NDArray db = delta.sum(Axis.ROW)
                        .divi(input.numCols());
      v.muli(0.9f).subi(dw.muli((float) learningRate));
      weights.addi(v);
      bias.subi(db.muli((float) learningRate));
      l1Update(learningRate, iteration);
      return dzOut;
   }

   @Override
   public Layer copy() {
      return new OutputLayer(this);
   }

   public static class Builder extends WeightLayerBuilder<Builder> {

      @Override
      public Layer build() {
         boolean isBinary = getOutputSize() <= 2;
         return new OutputLayer(getInputSize(), isBinary ? 1 : getOutputSize(), getActivation(), this.getInitializer(),
                                getL1(),
                                getL2());
      }
   }

}// END OF OutputLayer
