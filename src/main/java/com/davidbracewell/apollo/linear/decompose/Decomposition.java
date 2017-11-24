package com.davidbracewell.apollo.linear.decompose;

import com.davidbracewell.apollo.linear.NDArray;

/**
 * @author David B. Bracewell
 */
public interface Decomposition {

   NDArray[] decompose(NDArray input);

}//END OF Decomposition
