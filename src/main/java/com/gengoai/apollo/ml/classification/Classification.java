package com.gengoai.apollo.ml.classification;

import com.gengoai.apollo.linear.Axis;
import com.gengoai.apollo.linear.NDArray;
import com.gengoai.apollo.ml.vectorizer.Vectorizer;
import com.gengoai.collection.counter.Counter;
import com.gengoai.collection.counter.Counters;

import java.io.Serializable;

/**
 * The type Classification.
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
    * Instantiates a new Classification.
    *
    * @param distribution the distribution
    * @param vectorizer   the vectorizer
    */
   public Classification(NDArray distribution, Vectorizer<String> vectorizer) {
      this.distribution = distribution.isColumnVector() ? distribution.T() : distribution.copy();
      this.vectorizer = vectorizer;
   }

   /**
    * Arg max int.
    *
    * @return the int
    */
   public int argMax() {
      return (int) distribution.argMax(Axis.ROW).get(0);
   }

   /**
    * As counter counter.
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
    * Distribution double [ ].
    *
    * @return the double [ ]
    */
   public double[] distribution() {
      return distribution.toDoubleArray();
   }

   /**
    * Gets vectorizer.
    *
    * @return the vectorizer
    */
   public Vectorizer<String> getVectorizer() {
      return vectorizer;
   }

   /**
    * Sets vectorizer.
    *
    * @param vectorizer the vectorizer
    */
   public void setVectorizer(Vectorizer<String> vectorizer) {
      this.vectorizer = vectorizer;
   }

   /**
    * Select int [ ].
    *
    * @param minValue the min value
    * @return the int [ ]
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
