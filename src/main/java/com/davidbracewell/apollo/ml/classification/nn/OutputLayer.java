package com.davidbracewell.apollo.ml.classification.nn;

import com.davidbracewell.apollo.linalg.Matrix;
import com.davidbracewell.apollo.optimization.WeightInitializer;
import com.davidbracewell.apollo.optimization.activation.Activation;
import com.davidbracewell.apollo.optimization.activation.SigmoidActivation;
import com.davidbracewell.apollo.optimization.activation.SoftmaxActivation;
import com.davidbracewell.tuple.Tuple2;
import lombok.val;

/**
 * @author David B. Bracewell
 */
public class OutputLayer extends WeightLayer {
   public OutputLayer(int inputSize, int outputSize, Activation activation, WeightInitializer weightInitializer, double l1, double l2) {
      super(inputSize, outputSize, activation, weightInitializer, l1, l2);
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
   public Tuple2<Matrix, Double> backward(WeightUpdate updater, Matrix input, Matrix output, Matrix delta, int iteration, boolean calcuateDelta) {
      return updater.update(this.weights, this.bias, input, output, delta, iteration, calcuateDelta);
   }

   @Override
   public BackpropResult backward(Matrix input, Matrix output, Matrix delta, boolean calculateDelta) {
      Matrix dzOut = calculateDelta
                     ? weights.transpose().mmul(delta)
                     : null;
      val dw = delta.mmul(input.transpose());
      val db = delta.rowSums();
      return BackpropResult.from(dzOut, dw, db);
   }

   @Override
   public Matrix backward(Matrix input, Matrix output, Matrix delta, double learningRate, int layerIndex, int iteration) {
      Matrix dzOut = layerIndex > 0
                     ? weights.transpose().mmul(delta)
                     : null;
      val dw = delta.mmul(input.transpose())
                    .divi(input.numCols());
      val db = delta.rowSums()
                    .divi(input.numCols());
      v.muli(0.9).subi(dw.muli(learningRate));
      weights.addi(v);
      bias.subi(db.muli(learningRate));
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
         return new OutputLayer(getInputSize(), getOutputSize(), getActivation(), getWeightInitializer(), getL1(),
                                getL2());
      }
   }

}// END OF OutputLayer
