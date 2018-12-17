package com.gengoai.apollo.ml.vectorizer;

import com.gengoai.apollo.ml.Dataset;

/**
 * <p>Specialized vectorizer to encode binary classes (0/1).</p>
 *
 * @author David B. Bracewell
 */
public class BinaryVectorizer extends StringVectorizer {
   private static final long serialVersionUID = 1L;
   private final String trueLabel;
   private final String falseLabel;

   /**
    * Instantiates a new Binary vectorizer using "true" and "false" as the labels.
    */
   public BinaryVectorizer() {
      this("true", "false");
   }

   /**
    * Instantiates a new Binary vectorizer.
    *
    * @param trueLabel  the true label
    * @param falseLabel the false label
    */
   public BinaryVectorizer(String trueLabel, String falseLabel) {
      this.trueLabel = trueLabel;
      this.falseLabel = falseLabel;
   }

   @Override
   public boolean isLabelVectorizer() {
      return true;
   }

   @Override
   public String decode(double value) {
      return value == 1.0
             ? trueLabel
             : falseLabel;
   }

   @Override
   public double encode(String value) {
      return value.equals(trueLabel)
             ? 1.0
             : 0.0;
   }

   @Override
   public void fit(Dataset dataset) {

   }

   @Override
   public int size() {
      return 2;
   }

}//END OF BinaryVectorizer
