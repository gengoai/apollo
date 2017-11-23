package com.davidbracewell.apollo.ml.optimization.activation;

import com.davidbracewell.apollo.linear.NDArray;
import lombok.NonNull;

import java.io.Serializable;

/**
 * The interface Activation.
 *
 * @author David B. Bracewell
 */
public interface Activation extends Serializable {
   Activation LINEAR = new LinearActivation();
   Activation RELU = new ReLuActivation();
   Activation SIGMOID = new SigmoidActivation();
   Activation SOFTMAX = new SoftmaxActivation();

   /**
    * Apply double.
    *
    * @param x the x
    * @return the double
    */
   double apply(double x);

   /**
    * Apply vector.
    *
    * @param x the x
    * @return the vector
    */
   default NDArray apply(@NonNull NDArray x) {
      return x.mapi(this::apply);
   }


   /**
    * Gradient vector.
    *
    * @param in the in
    * @return the vector
    */
   default NDArray gradient(@NonNull NDArray in) {
      return valueGradient(apply(in));
   }

   /**
    * Gradient double.
    *
    * @param in the in
    * @return the double
    */
   default double gradient(double in) {
      return valueGradient(apply(in));
   }

   /**
    * Is probabilistic boolean.
    *
    * @return the boolean
    */
   default boolean isProbabilistic() {
      return false;
   }

   /**
    * Value gradient double.
    *
    * @param activated the activated
    * @return the double
    */
   double valueGradient(double activated);

   /**
    * Value gradient vector.
    *
    * @param activated the activated
    * @return the vector
    */
   default NDArray valueGradient(@NonNull NDArray activated) {
      return activated.map(this::valueGradient);
   }

}//END OF Activation
