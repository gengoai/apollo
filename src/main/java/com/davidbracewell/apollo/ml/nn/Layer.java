package com.davidbracewell.apollo.ml.nn;

import com.davidbracewell.apollo.linalg.Vector;

/**
 * @author David B. Bracewell
 */
public interface Layer {

   Vector backward(Vector predicted, Vector actual);

   Layer connect(Layer source);

   Vector forward(Vector m);

   int getInputDimension();

   Layer setInputDimension(int dimension);

   int getOutputDimension();

}//END OF Layer
