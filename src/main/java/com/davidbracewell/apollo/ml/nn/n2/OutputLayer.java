package com.davidbracewell.apollo.ml.nn.n2;

import com.davidbracewell.apollo.linalg.DenseMatrix;
import com.davidbracewell.apollo.linalg.SparseVector;
import com.davidbracewell.apollo.linalg.Vector;
import com.davidbracewell.apollo.optimization.Weights;
import com.davidbracewell.apollo.optimization.activation.Activation;
import com.davidbracewell.apollo.optimization.activation.SoftmaxActivation;
import com.davidbracewell.apollo.optimization.update.WeightUpdate;

/**
 * @author David B. Bracewell
 */
public class OutputLayer implements Layer {

   private final Activation activation;
   private final int outputSize;
   private Weights weights;
   private int inputSize;

   public OutputLayer(Activation activation, int nOut) {
      this.activation = activation;
      this.outputSize = nOut;
   }

   public OutputLayer(int nOut) {
      this(new SoftmaxActivation(), nOut);
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
   public void connect(Layer previousLayer) {
      this.inputSize = previousLayer.getOutputSize();
      double wr = 1.0 / Math.sqrt(6.0 / outputSize + inputSize);
      this.weights = new Weights(DenseMatrix.random(outputSize, inputSize, -wr, wr),
                                 SparseVector.zeros(outputSize),
                                 false
      );
   }

   @Override
   public Vector forward(Vector input) {
      return activation.apply(this.weights.dot(input));
   }

   @Override
   public int getInputSize() {
      return inputSize;
   }

   @Override
   public int getOutputSize() {
      return outputSize;
   }

}// END OF OutputLayer
