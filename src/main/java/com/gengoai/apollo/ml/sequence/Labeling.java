package com.gengoai.apollo.ml.sequence;

import java.io.Serializable;

/**
 * <p>Represents the predicted labels for a sequence.</p>
 *
 * @author David B. Bracewell
 */
public class Labeling implements Serializable {
   private final String[] labels;


   public Labeling(String[] labels) {
      this.labels = labels;
   }

   /**
    * Gets the label for the item in the sequence at the given index.
    *
    * @param index the index of the item in the sequence
    * @return the int label
    */
   public String getLabel(int index) {
      return labels[index];
   }

}//END OF Labeling
