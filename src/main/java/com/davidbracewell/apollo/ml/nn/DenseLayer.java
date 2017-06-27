package com.davidbracewell.apollo.ml.nn;

import com.davidbracewell.apollo.linalg.DenseMatrix;
import com.davidbracewell.apollo.linalg.Matrix;

/**
 * @author David B. Bracewell
 */
public class DenseLayer implements Layer {
   private final int outputDimension;
   private int inputDimension;
   private Matrix weights;

   public DenseLayer(int outputDimension) {
      this.outputDimension = outputDimension;
   }

   @Override
   public Matrix backward(Matrix m) {
      Matrix error = m.dot(weights.T());
      return error;
   }

   @Override
   public Layer connect(Layer source) {
      this.inputDimension = source.getOutputDimension();
      return this;
   }

   @Override
   public Matrix forward(Matrix m) {
      return m.multiply(weights);
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
      this.weights = new DenseMatrix(inputDimension, outputDimension);
      return this;
   }
}// END OF DenseLayer
