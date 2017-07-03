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
    * @param numComponents the num components
    * @param initializer   the initializer
    */
   public WeightComponent(int numComponents, @NonNull WeightInitializer initializer) {
      Preconditions.checkArgument(numComponents > 0, "Need at least one weight component");
      this.weights = new Weights[numComponents];
      for (Weights weight : this.weights) {
         initializer.initialize(weight.getTheta());
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
