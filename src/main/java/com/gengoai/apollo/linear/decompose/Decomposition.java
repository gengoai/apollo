package com.gengoai.apollo.linear.decompose;

import com.gengoai.apollo.linear.NDArray;
import com.gengoai.apollo.linear.Tensor;

import java.io.Serializable;

/**
 * <p>Encapsulates an algorithm to decompose (factorize) a given {@link NDArray} into a product of matrices.</p>
 *
 * @author David B. Bracewell
 */
public abstract class Decomposition implements Serializable {
   private static final long serialVersionUID = 1L;
   final int components;

   protected Decomposition(int components) {
      this.components = components;
   }


   public int getNumberOfComponents() {
      return components;
   }

   protected abstract NDArray[] onMatrix(NDArray matrix);

   /**
    * Decompose the given input NDArray into a product of one or more other NDArrays
    *
    * @param input the input NDArray
    * @return Array of NDArray representing the factors of the product.
    */
   public final NDArray[] decompose(NDArray input) {
      if (input.shape().order() < 3) {
         return onMatrix(input);
      }
      NDArray[][] results = new NDArray[components][input.shape().sliceLength];
      for (int i = 0; i < input.shape().sliceLength; i++) {
         NDArray[] slice = onMatrix(input.slice(i));
         for (int j = 0; j < components; j++) {
            results[j][i] = slice[j];
         }
      }
      NDArray[] out = new NDArray[components];
      for (int j = 0; j < components; j++) {
         out[j] = new Tensor(input.kernels(), input.channels(), results[j]);
      }
      return out;
   }

}//END OF Decomposition
