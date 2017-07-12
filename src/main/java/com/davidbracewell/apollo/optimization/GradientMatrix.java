package com.davidbracewell.apollo.optimization;

import com.davidbracewell.apollo.linalg.Vector;
import lombok.NonNull;

import java.io.Serializable;

/**
 * The type Gradient matrix.
 *
 * @author David B. Bracewell
 */
public class GradientMatrix implements Serializable {
   private static final long serialVersionUID = 1L;
   private final Gradient[] gradients;

   /**
    * Instantiates a new Gradient matrix.
    *
    * @param size the size
    */
   public GradientMatrix(int size) {
      this.gradients = new Gradient[size];
   }


   /**
    * Calculate gradient matrix.
    *
    * @param input the input
    * @param error the error
    * @return the gradient matrix
    */
   public static GradientMatrix calculate(@NonNull Vector input, @NonNull Vector error) {
      GradientMatrix gm = new GradientMatrix(error.dimension());
      for (int i = 0; i < gm.size(); i++) {
         gm.set(i, Gradient.of(input.mapMultiply(error.get(i)), error.get(i)));
      }
      return gm;
   }

   public GradientMatrix add(@NonNull GradientMatrix other) {
      for (int i = 0; i < gradients.length; i++) {
         gradients[i].getWeightGradient().addSelf(other.gradients[i].getWeightGradient());
         gradients[i].setBiasGradient(gradients[i].getBiasGradient() + other.gradients[i].getBiasGradient());
      }
      return this;
   }

   /**
    * Get gradient.
    *
    * @param i the
    * @return the gradient
    */
   public Gradient get(int i) {
      return gradients[i];
   }

   /**
    * Scale gradient matrix.
    *
    * @param value the value
    * @return the gradient matrix
    */
   public GradientMatrix scale(double value) {
      for (Gradient gradient : gradients) {
         gradient.scale(value);
      }
      return this;
   }

   /**
    * Set.
    *
    * @param i        the
    * @param gradient the gradient
    */
   public void set(int i, @NonNull Gradient gradient) {
      gradients[i] = gradient;
   }

   /**
    * Size int.
    *
    * @return the int
    */
   public int size() {
      return gradients.length;
   }

}// END OF GradientMatrix
