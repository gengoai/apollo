package com.davidbracewell.apollo.optimization;

import lombok.NonNull;

import java.io.Serializable;
import java.util.Arrays;
import java.util.Collection;
import java.util.Iterator;

/**
 * The type Weight component.
 *
 * @author David B. Bracewell
 */
public class WeightComponent implements Serializable, Iterable<Weights> {
   private static final long serialVersionUID = 1L;
   private final Weights[] weights;

   public WeightComponent(Collection<Weights> weights) {
      this.weights = weights.toArray(new Weights[weights.size()]);
   }

   public WeightComponent(Weights weights) {
      this.weights = new Weights[1];
      this.weights[0] = weights;
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
