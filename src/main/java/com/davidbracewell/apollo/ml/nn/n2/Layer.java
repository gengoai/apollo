package com.davidbracewell.apollo.ml.nn.n2;

import com.davidbracewell.apollo.linalg.Vector;
import com.davidbracewell.apollo.optimization.update.WeightUpdate;

/**
 * The interface Layer.
 *
 * @author David B. Bracewell
 */
public interface Layer {

   /**
    * Backward vector.
    *
    * @param input        the input
    * @param output       the output
    * @param delta        the delta
    * @param weightUpdate the weight update
    * @param lr           the lr
    * @return the vector
    */
   Vector backward(Vector input, Vector output, Vector delta, WeightUpdate weightUpdate, double lr);

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


}//END OF Layer
