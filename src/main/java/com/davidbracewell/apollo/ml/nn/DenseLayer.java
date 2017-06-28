package com.davidbracewell.apollo.ml.nn;

import com.davidbracewell.apollo.linalg.DenseMatrix;
import com.davidbracewell.apollo.linalg.Matrix;
import com.davidbracewell.apollo.linalg.Vector;
import com.davidbracewell.apollo.optimization.activation.DifferentiableActivation;

/**
 * @author David B. Bracewell
 */
public class DenseLayer implements Layer {
   private final DifferentiableActivation activation;
   private final int outputDimension;
   private int inputDimension;
   private Matrix weights;

   public DenseLayer(DifferentiableActivation activation, int outputDimension) {
      this.activation = activation;
      this.outputDimension = outputDimension;
   }

   @Override
   public Vector calculateGradient(Vector activatedInput) {
      return activation.valueGradient(activatedInput);
   }

   @Override
   public Layer connect(Layer source) {
      setInputDimension(source.getOutputDimension());
      return this;
   }

   @Override
   public Vector forward(Vector m) {
      return activation.apply(weights.dot(m).column(0));
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
   public Matrix getWeights() {
      return weights;
   }

   @Override
   public Layer setInputDimension(int dimension) {
      this.inputDimension = dimension;
      double wr = 1.0 / Math.sqrt(6.0 / outputDimension + dimension);
      this.weights = DenseMatrix.random(outputDimension, inputDimension, -wr, wr);
      return this;
   }
}// END OF DenseLayer
