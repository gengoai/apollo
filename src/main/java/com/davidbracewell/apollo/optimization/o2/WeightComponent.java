package com.davidbracewell.apollo.optimization.o2;

import com.davidbracewell.apollo.optimization.Weights;
import com.davidbracewell.guava.common.base.Preconditions;
import lombok.NonNull;

import java.io.Serializable;
import java.util.Arrays;
import java.util.Iterator;

/**
 * The type Weight component.
 *
 * @author David B. Bracewell
 */
public class WeightComponent implements Serializable, Iterable<Weights> {
   private static final long serialVersionUID = 1L;
   private final Weights[] weights;

   /**
    * Instantiates a new Weight component.
    *
    * @param initializer the initializer
    */
   public WeightComponent(int[][] shapes, @NonNull WeightInitializer initializer) {
      Preconditions.checkArgument(shapes.length > 0, "Need at least one weight component");
      this.weights = new Weights[shapes.length];
      for (int i = 0; i < this.weights.length; i++) {
         int[] shape = shapes[i];
         this.weights[i] = shape[0] <= 2 ? Weights.binary(shape[1]) : Weights.multiClass(shape[0], shape[1]);
         initializer.initialize(this.weights[i].getTheta());
      }
   }

   /**
    * Get weights.
    *
    * @param index the index
    * @return the weights
    */
   public Weights get(int index) {
      return weights[index];
   }

   @Override
   public Iterator<Weights> iterator() {
      return Arrays.asList(weights).iterator();
   }

   /**
    * Set.
    *
    * @param index the index
    * @param w     the w
    */
   public void set(int index, @NonNull Weights w) {
      weights[index] = w;
   }

   /**
    * Size int.
    *
    * @return the int
    */
   public int size() {
      return weights.length;
   }

}// END OF WeightComponent
