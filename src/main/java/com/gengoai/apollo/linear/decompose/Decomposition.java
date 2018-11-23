package com.gengoai.apollo.linear.decompose;

import com.gengoai.apollo.linear.NDArray;

/**
 * <p>Encapsulates an algorithm to decompose (factorize) a given {@link NDArray} into a product of matrices.</p>
 *
 * @author David B. Bracewell
 */
public interface Decomposition {

   /**
    * Decompose the given input NDArray into a product of one or more other NDArrays
    *
    * @param input the input NDArray
    * @return Array of NDArray representing the factors of the product.
    */
   NDArray[] decompose(NDArray input);

}//END OF Decomposition
