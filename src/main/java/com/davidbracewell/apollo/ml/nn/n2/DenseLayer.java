package com.davidbracewell.apollo.ml.nn.n2;

import com.davidbracewell.apollo.linalg.DenseMatrix;
import com.davidbracewell.apollo.linalg.SparseVector;
import com.davidbracewell.apollo.linalg.Vector;
import com.davidbracewell.apollo.optimization.Weights;
import com.davidbracewell.apollo.optimization.update.WeightUpdate;

/**
 * @author David B. Bracewell
 */
public class DenseLayer implements Layer {
   private final int outputSize;
   private Weights weights;
   private int inputSize;

   public DenseLayer(int nOut) {
      this.outputSize = nOut;
   }

   @Override
   public Vector backward(Vector input, Vector output, Vector delta, WeightUpdate weightUpdate, double lr) {
      Vector dPrime = delta.toMatrix()
                           .multiply(weights.getTheta())
                           .row(0);
      weightUpdate.update(weights,
                          new Weights(delta.transpose().multiply(input.toMatrix()), delta, false),
                          lr);
      return dPrime;
   }

   @Override
   public Vector forward(Vector input) {
      return this.weights.dot(input);
   }

   @Override
   public int getInputSize() {
      return inputSize;
   }

   @Override
   public int getOutputSize() {
      return outputSize;
   }

   @Override
   public void connect(Layer previousLayer) {
      this.inputSize = previousLayer.getOutputSize();
      this.weights = new Weights(DenseMatrix.random(outputSize, inputSize),
                                 SparseVector.zeros(outputSize),
                                 false
      );
   }
}// END OF DenseLayer
