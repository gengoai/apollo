package com.gengoai.apollo.ml;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;

/**
 * The type Labeled sequence.
 *
 * @param <T> the type parameter
 * @author David B. Bracewell
 */
public class LabeledSequence<T> implements Serializable, Iterable<LabeledDatum<T>> {
   private static final long serialVersionUID = 1L;
   private List<LabeledDatum<T>> sequence = new ArrayList<>();


   /**
    * Instantiates a new Labeled sequence.
    */
   public LabeledSequence() {

   }

   public LabeledSequence(List<T> data) {
      for (T datum : data) {
         sequence.add(LabeledDatum.of(null, datum));
      }
   }

   /**
    * Instantiates a new Labeled sequence.
    *
    * @param instances the instances
    */
   @SafeVarargs
   public LabeledSequence(LabeledDatum<T>... instances) {
      Collections.addAll(sequence, instances);
   }

   /**
    * Add.
    *
    * @param datum the datum
    */
   public void add(LabeledDatum<T> datum) {
      sequence.add(datum);
   }

   /**
    * Add.
    *
    * @param label the label
    * @param data  the data
    */
   public void add(Object label, T data) {
      sequence.add(LabeledDatum.of(label, data));
   }


   @Override
   public Iterator<LabeledDatum<T>> iterator() {
      return sequence.iterator();
   }


   /**
    * Get labeled datum.
    *
    * @param index the index
    * @return the labeled datum
    */
   public LabeledDatum<T> get(int index) {
      return sequence.get(index);
   }

   /**
    * Size int.
    *
    * @return the int
    */
   public int size() {
      return sequence.size();
   }

}//END OF LabeledSequence
