package com.davidbracewell.apollo.ml.nn;

import com.davidbracewell.apollo.linalg.Matrix;

/**
 * @author David B. Bracewell
 */
public interface Layer {

   Matrix forward(Matrix input);

}//END OF Layer
