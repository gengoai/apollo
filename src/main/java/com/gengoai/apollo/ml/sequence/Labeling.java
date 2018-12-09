package com.gengoai.apollo.ml.sequence;

import com.gengoai.apollo.linear.NDArray;
import com.gengoai.apollo.ml.vectorizer.Vectorizer;

import java.io.Serializable;

/**
 * @author David B. Bracewell
 */
public class Labeling implements Serializable {
   private final double[] labels;
   private final Vectorizer<String> vectorizer;


   public Labeling(NDArray labeling) {
      this(labeling, null);
   }

   public Labeling(NDArray labeling, Vectorizer<String> vectorizer) {
      this.labels = labeling.toDoubleArray();
      this.vectorizer = vectorizer;
   }

   public double get(int index) {
      return labels[index];
   }

   public String getString(int index) {
      return vectorizer.decode(labels[index]);
   }

}//END OF Labeling
