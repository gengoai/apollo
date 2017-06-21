package com.davidbracewell.apollo.ml.nn;

import com.davidbracewell.apollo.linalg.DenseMatrix;
import com.davidbracewell.apollo.linalg.Matrix;
import com.davidbracewell.apollo.linalg.Vector;
import com.davidbracewell.apollo.optimization.activation.Activation;
import lombok.NonNull;

import java.io.Serializable;

/**
 * @author David B. Bracewell
 */
public class HiddenLayer implements Layer, Serializable {
   private static final long serialVersionUID = 1L;
   private final Activation activation;
   private int nOut;
   private Matrix weights;
   private Vector bias;

   public HiddenLayer(@NonNull Activation activation) {
      this(0, activation);
   }

   public HiddenLayer(int nOut, @NonNull Activation activation) {
      this.activation = activation;
   }

   @Override
   public Matrix forward(Matrix input) {
      return input.multiply(weights)
                  .addRowSelf(bias)
                  .mapRowSelf(activation::apply);
   }

   @Override
   public int getInputSize() {
      return weights == null ? 0 : weights.numberOfRows();
   }

   @Override
   public int getOutputSize() {
      return nOut;
   }

   public void init(int nIn) {
      this.weights = DenseMatrix.random(nOut, nIn);
      this.bias = Vector.sZeros(nOut);
   }

   @Override
   public void init(int nIn, int nOut) {
      this.nOut = nOut;
      init(nIn);
   }

   @Override
   public void reset() {
      this.weights = null;
      this.bias = null;
   }

}// END OF HiddenLayer
