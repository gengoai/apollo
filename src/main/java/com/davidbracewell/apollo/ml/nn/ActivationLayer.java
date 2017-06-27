package com.davidbracewell.apollo.ml.nn;

import com.davidbracewell.apollo.linalg.Matrix;
import com.davidbracewell.apollo.optimization.activation.DifferentiableActivation;
import lombok.NonNull;

/**
 * @author David B. Bracewell
 */
public class ActivationLayer implements Layer {
   private final DifferentiableActivation activation;
   private int outputDimension;
   private int inputDimension;
   private Matrix forwardResult = null;

   public ActivationLayer(@NonNull DifferentiableActivation activation) {
      this.activation = activation;
   }

   @Override
   public Matrix backward(Matrix m) {
      return activation.valueGradient(forwardResult, m);
   }

   @Override
   public Layer connect(Layer source) {
      this.inputDimension = source.getInputDimension();
      this.outputDimension = source.getOutputDimension();
      return this;
   }

   @Override
   public Matrix forward(Matrix m) {
      this.inputDimension = m.numberOfRows();
      this.outputDimension = m.numberOfColumns();
      this.forwardResult = m.mapRow(activation::apply);
      return this.forwardResult;
   }

   @Override
   public int getInputDimension() {
      return inputDimension;
   }

   @Override
   public int getOutputDimension() {
      return outputDimension;
   }

   @Override
   public Layer setInputDimension(int dimension) {
      this.inputDimension = dimension;
      return this;
   }

}// END OF ActivationLayer
