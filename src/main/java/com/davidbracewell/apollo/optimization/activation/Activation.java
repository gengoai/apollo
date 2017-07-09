package com.davidbracewell.apollo.optimization.activation;

import com.davidbracewell.apollo.linalg.Matrix;
import com.davidbracewell.apollo.linalg.Vector;
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
   default Vector apply(@NonNull Vector x) {
      return x.map(this::apply);
   }


   /**
    * Apply matrix.
    *
    * @param m the m
    * @return the matrix
    */
   default Matrix apply(@NonNull Matrix m) {
      return m.mapRow(this::apply);
   }

   /**
    * Gradient vector.
    *
    * @param in the in
    * @return the vector
    */
   default Vector gradient(@NonNull Vector in) {
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
   default Vector valueGradient(@NonNull Vector activated) {
      return activated.map(this::valueGradient);
   }

}//END OF Activation
