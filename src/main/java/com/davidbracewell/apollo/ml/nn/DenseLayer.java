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
   public Vector backward(Vector predicted, Vector actual) {
      Vector delta = activation.valueGradient(predicted, actual);
      System.out.println(delta.dimension() + " :" + weights.shape());
      weights.addColumnSelf(delta);
      return delta;
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
   public Layer setInputDimension(int dimension) {
      this.inputDimension = dimension;
      this.weights = new DenseMatrix(outputDimension, inputDimension);
      return this;
   }
}// END OF DenseLayer
