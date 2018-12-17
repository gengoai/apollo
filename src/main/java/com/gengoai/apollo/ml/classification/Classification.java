package com.gengoai.apollo.ml.classification;

import com.gengoai.apollo.linear.Axis;
import com.gengoai.apollo.linear.NDArray;
import com.gengoai.apollo.ml.vectorizer.Vectorizer;
import com.gengoai.collection.counter.Counter;
import com.gengoai.collection.counter.Counters;

import java.io.Serializable;

/**
 * Encapsulates the result of a classifier model applied to an instance.
 *
 * @author David B. Bracewell
 */
public class Classification implements Serializable {
   private static final long serialVersionUID = 1L;
   private final NDArray distribution;
   private Vectorizer<String> vectorizer;


   /**
    * Instantiates a new Classification.
    *
    * @param distribution the distribution
    */
   public Classification(NDArray distribution) {
      this(distribution, null);
   }

   /**
    * Instantiates a new Classification with a vectorizer to facilitate label id to label mapping.
    *
    * @param distribution the distribution
    * @param vectorizer   the vectorizer
    */
   public Classification(NDArray distribution, Vectorizer<String> vectorizer) {
      this.distribution = distribution.isColumnVector() ? distribution.T() : distribution.copy();
      this.vectorizer = vectorizer;
   }

   /**
    * The label id with maximum score.
    *
    * @return the label id with the maximum score.
    */
   public int argMax() {
      return (int) distribution.argMax(Axis.ROW).get(0);
   }

   /**
    * The label id with minimum score.
    *
    * @return the label id with the minimum score.
    */
   public int argMin() {
      return (int) distribution.argMin(Axis.ROW).get(0);
   }


   /**
    * Gets the classification object as a Counter. Will convert to label ids to names if a vectorizer is present,
    * otherwise will use string representation of label ids.
    *
    * @return the counter
    */
   public Counter<String> asCounter() {
      Counter<String> counter = Counters.newCounter();
      for (long i = 0; i < distribution.length(); i++) {
         counter.set((vectorizer == null ? Long.toString(i) : vectorizer.decode(i)),
                     distribution.get((int) i));
      }
      return counter;
   }

   /**
    * Gets the underlying distribution of scores.
    *
    * @return the NDArray representing the distribution.
    */
   public NDArray distribution() {
      return distribution;
   }

   /**
    * Gets the argMax as a string either converting the id using the supplied vectorizer or using
    * <code>Integer.toString</code>
    *
    * @return the result
    */
   public String getResult() {
      if (vectorizer == null) {
         return Integer.toString(argMax());
      }
      return vectorizer.decode(argMax());
   }

   /**
    * Selects all label ids whose score is greater than or equal to the given minimum value.
    *
    * @param minValue the minimum value
    * @return the array of integers with label ids whose score is greater than or equal to the given minimum value.
    */
   public int[] select(double minValue) {
      NDArray temp = distribution.test(v -> v >= minValue);
      int[] rval = new int[(int) temp.scalarSum()];
      for (long i = 0, j = 0; i < temp.length(); i++) {
         if (temp.get((int) i) == 1) {
            rval[(int) j] = (int) i;
            j++;
         }
      }
      return rval;
   }

   @Override
   public String toString() {
      return "Classification" + distribution;

   }
}//END OF Classification
