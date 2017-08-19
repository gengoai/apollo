package com.davidbracewell.apollo.ml.classification.nn.slt;

import com.davidbracewell.apollo.linalg.Matrix;
import com.davidbracewell.apollo.optimization.WeightInitializer;
import com.davidbracewell.apollo.optimization.activation.Activation;
import com.davidbracewell.apollo.optimization.activation.SigmoidActivation;
import com.davidbracewell.apollo.optimization.activation.SoftmaxActivation;
import lombok.val;

/**
 * @author David B. Bracewell
 */
public class OutputLayer extends WeightLayer {
   public OutputLayer(int inputSize, int outputSize, Activation activation, WeightInitializer weightInitializer, double l1, double l2) {
      super(inputSize, outputSize, activation, weightInitializer, l1, l2);
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
   public Matrix backward(Matrix input, Matrix output, Matrix delta, double learningRate, int layerIndex) {
      Matrix dzOut = layerIndex > 0
                     ? weights.transpose().mmul(delta)
                     : null;
      val dw = delta.mmul(input.transpose())
                    .divi(input.numCols());
      val db = delta.rowSums()
                    .divi(input.numCols());
      weights.subi(dw.muli(learningRate));
      bias.subi(db.muli(learningRate));
      return dzOut;
   }

   public static class Builder extends WeightLayerBuilder<Builder> {

      @Override
      public Layer build() {
         return new OutputLayer(getInputSize(), getOutputSize(), getActivation(), getWeightInitializer(),getL1(),getL2());
      }
   }

}// END OF OutputLayer
