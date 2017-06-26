package com.davidbracewell.apollo.ml.nn;

import com.davidbracewell.apollo.linalg.Matrix;

/**
 * @author David B. Bracewell
 */
public interface Layer {

   Matrix backward(Matrix m);

   Matrix forward(Matrix m);

}//END OF Layer
