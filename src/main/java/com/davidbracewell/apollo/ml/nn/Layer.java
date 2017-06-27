package com.davidbracewell.apollo.ml.nn;

import com.davidbracewell.apollo.linalg.Matrix;

/**
 * @author David B. Bracewell
 */
public interface Layer {

   Matrix backward(Matrix m);

   Matrix forward(Matrix m);

   Layer setInputDimension(int dimension);

   int getInputDimension();

   int getOutputDimension();

   Layer connect(Layer source);

}//END OF Layer
