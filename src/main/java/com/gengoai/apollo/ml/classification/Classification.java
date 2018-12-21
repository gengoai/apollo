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
   private final String argMax;
   private final NDArray distribution;
   private Vectorizer<String> vectorizer;

   /**
    * Instantiates a new Classification with a vectorizer to facilitate label id to label mapping.
    *
    * @param distribution the distribution
    * @param vectorizer   the vectorizer
    */
   public Classification(NDArray distribution, Vectorizer<String> vectorizer) {
      this.distribution = distribution.isColumnVector() ? distribution.T() : distribution.copy();
      this.argMax = vectorizer.decode(this.distribution.argMax(Axis.ROW).scalarValue());
      this.vectorizer = vectorizer;
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
         counter.set(vectorizer.decode(i), distribution.get((int) i));
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


   public double getScore(String label){
      return distribution.get((int)vectorizer.encode(label));
   }

   /**
    * Gets the argMax as a string either converting the id using the supplied vectorizer or using
    * <code>Integer.toString</code>
    *
    * @return the result
    */
   public String getResult() {
      return argMax;
   }

   @Override
   public String toString() {
      return "Classification" + distribution;

   }
}//END OF Classification
