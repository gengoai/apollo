package com.gengoai.apollo.ml.sequence;

import lombok.NonNull;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Optional;

/**
 * The type Sequence input.
 *
 * @param <T> the type parameter
 * @author David B. Bracewell
 */
public class SequenceInput<T> implements Serializable {
   private static final long serialVersionUID = 1L;
   private final ArrayList<T> list = new ArrayList<>();
   private final ArrayList<String> labels = new ArrayList<>();

   /**
    * Instantiates a new Sequence input.
    */
   public SequenceInput() {

   }

   /**
    * Instantiates a new Sequence input.
    *
    * @param collection the collection
    */
   public SequenceInput(@NonNull Collection<T> collection) {
      list.addAll(collection);
   }


   /**
    * Gets label.
    *
    * @param index the index
    * @return the label
    */
   public String getLabel(int index) {
      if (index < 0 || index >= labels.size()) {
         return null;
      }
      return labels.get(index);
   }

   public void setLabel(int index, String label) {
      labels.set(index, label);
   }

   /**
    * Get t.
    *
    * @param index the index
    * @return the t
    */
   public T get(int index) {
      if (index < 0 || index >= list.size()) {
         return null;
      }
      return list.get(index);
   }

   /**
    * Add.
    *
    * @param observation the observation
    * @param label       the label
    */
   public void add(T observation, String label) {
      list.add(observation);
      labels.add(label);
   }

   /**
    * Add.
    *
    * @param observation the observation
    */
   public void add(T observation) {
      list.add(observation);
      labels.add(null);
   }

   /**
    * Size int.
    *
    * @return the int
    */
   public int size() {
      return list.size();
   }

   /**
    * Iterator contextual iterator.
    *
    * @return the contextual iterator
    */
   public Context<T> iterator() {
      return new Itr();
   }

   private class Itr extends Context<T> {
      private static final long serialVersionUID = 1L;

      @Override
      public int size() {
         return SequenceInput.this.size();
      }

      @Override
      protected Optional<T> getContextAt(int index) {
         return Optional.ofNullable(SequenceInput.this.get(index));
      }

      @Override
      protected Optional<String> getLabelAt(int index) {
         return Optional.ofNullable(SequenceInput.this.getLabel(index));
      }
   }

}// END OF SequenceInput
