package com.gengoai.apollo.linear.decompose;

import com.gengoai.apollo.linear.NDArray;

/**
 * @author David B. Bracewell
 */
public interface Decomposition {

   NDArray[] decompose(NDArray input);

}//END OF Decomposition
