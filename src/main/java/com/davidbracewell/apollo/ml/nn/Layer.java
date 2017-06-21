package com.davidbracewell.apollo.ml.nn;

import com.davidbracewell.apollo.linalg.Matrix;

/**
 * @author David B. Bracewell
 */
public interface Layer {

   Matrix forward(Matrix input);

   int getInputSize();

   int getOutputSize();

   void init(int nIn);

   void init(int nIn, int nOut);

   void reset();

}//END OF Layer
