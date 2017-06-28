package com.davidbracewell.apollo.ml.nn;

import com.davidbracewell.apollo.linalg.Matrix;
import com.davidbracewell.apollo.linalg.Vector;

/**
 * @author David B. Bracewell
 */
public interface Layer {

   Vector calculateGradient(Vector activatedInput);

   Layer connect(Layer source);

   Vector forward(Vector m);

   int getInputDimension();

   Layer setInputDimension(int dimension);

   int getOutputDimension();

   Matrix getWeights();

}//END OF Layer
