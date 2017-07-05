package com.davidbracewell.apollo.ml.nn;

import com.davidbracewell.apollo.linalg.DenseMatrix;
import com.davidbracewell.apollo.linalg.DenseVector;
import com.davidbracewell.apollo.linalg.Vector;
import com.davidbracewell.apollo.optimization.Weights;

/**
 * The type Dense layer.
 *
 * @author David B. Bracewell
 */
public class DenseLayer implements Layer {
   private static final long serialVersionUID = 1L;
   private final int outputSize;
   private Weights weights;
   private int inputSize;

   /**
    * Instantiates a new Dense layer.
    *
    * @param nOut the n out
    */
   public DenseLayer(int nOut) {
      this.outputSize = nOut;
   }

   @Override
   public Vector backward(Vector output, Vector delta) {
      return delta.toMatrix()
                  .multiply(weights.getTheta())
                  .row(0);
   }

   @Override
   public void connect(Layer previousLayer) {
      this.inputSize = previousLayer.getOutputSize();
      this.weights = new Weights(DenseMatrix.zeroes(outputSize, inputSize),
                                 DenseVector.zeros(outputSize),
                                 outputSize <= 2
      );
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
   public Weights getWeights() {
      return weights;
   }

   @Override
   public void setWeights(Weights weights) {
      this.weights = weights;
   }

   @Override
   public boolean hasWeights() {
      return true;
   }
}// END OF DenseLayer
