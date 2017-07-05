package com.davidbracewell.apollo.ml.nn;

import com.davidbracewell.apollo.linalg.Vector;
import com.davidbracewell.apollo.optimization.Weights;
import com.davidbracewell.stream.MStream;

import java.io.Serializable;

/**
 * The interface Layer.
 *
 * @author David B. Bracewell
 */
public interface Layer extends Serializable {

   /**
    * Backward vector.
    *
    * @param output the output
    * @param delta  the delta
    * @return the vector
    */
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

   /**
    * Gets weights.
    *
    * @return the weights
    */
   default Weights getWeights() {
      return null;
   }

   /**
    * Sets weights.
    *
    * @param weights the weights
    */
   default void setWeights(Weights weights) {
   }

   /**
    * Has weights boolean.
    *
    * @return the boolean
    */
   boolean hasWeights();

   default MStream<Vector> preTrain(MStream<Vector> previousOutput, double learningRate) {
      return previousOutput;
   }

}//END OF Layer
