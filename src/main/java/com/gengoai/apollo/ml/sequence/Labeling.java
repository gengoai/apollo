package com.gengoai.apollo.ml.sequence;

import com.gengoai.apollo.linear.NDArray;
import com.gengoai.apollo.ml.vectorizer.Vectorizer;

import java.io.Serializable;

/**
 * <p>Represents the predicted labels for a sequence.</p>
 *
 * @author David B. Bracewell
 */
public class Labeling implements Serializable {
   private final double[] labels;
   private final Vectorizer<String> vectorizer;


   /**
    * Instantiates a new Labeling.
    *
    * @param labeling the labeling
    */
   public Labeling(NDArray labeling) {
      this(labeling, null);
   }

   /**
    * Instantiates a new Labeling with a given vectorizer to decode ids into string labels.
    *
    * @param labeling   the labeling
    * @param vectorizer the vectorizer
    */
   public Labeling(NDArray labeling, Vectorizer<String> vectorizer) {
      this.labels = labeling.toDoubleArray();
      this.vectorizer = vectorizer;
   }

   /**
    * Gets the label for the item in the sequence at the given index.
    *
    * @param index the index of the item in the sequence
    * @return the int label
    */
   public int getLabel(int index) {
      return (int) labels[index];
   }

   /**
    * Gets the label for the item in the sequence at the given index as a String.
    *
    * @param index the index of the item in the sequence
    * @return the String label
    */
   public String getStringLabel(int index) {
      if (vectorizer != null) {
         return vectorizer.decode(labels[index]);
      }
      return Integer.toString(getLabel(index));
   }

}//END OF Labeling
