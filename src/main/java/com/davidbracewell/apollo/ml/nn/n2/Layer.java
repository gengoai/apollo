package com.davidbracewell.apollo.ml.nn.n2;

import com.davidbracewell.apollo.linalg.Vector;
import com.davidbracewell.apollo.optimization.Weights;

/**
 * The interface Layer.
 *
 * @author David B. Bracewell
 */
public interface Layer {

   Vector backward(Vector output, Vector delta);

   /**
    * Connect.
    *
    * @param previousLayer the previous layer
    */
   void connect(Layer previousLayer);

   /**
    * Forward vector.
    *
    * @param input the input
    * @return the vector
    */
   Vector forward(Vector input);

   /**
    * Gets input size.
    *
    * @return the input size
    */
   int getInputSize();

   /**
    * Gets output size.
    *
    * @return the output size
    */
   int getOutputSize();

   default Weights getWeights() {
      return null;
   }

   default void setWeights(Weights weights) {
   }

   boolean isOptimizable();

}//END OF Layer
