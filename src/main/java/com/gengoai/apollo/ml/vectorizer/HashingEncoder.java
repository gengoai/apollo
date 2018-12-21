package com.gengoai.apollo.ml.vectorizer;

import com.gengoai.apollo.ml.data.Dataset;
import com.gengoai.json.JsonEntry;

import java.util.Collections;
import java.util.Set;

/**
 * @author David B. Bracewell
 */
public class HashingEncoder extends StringVectorizer {
   private static final long serialVersionUID = 1L;
   private final boolean isBinary;
   private final int numberOfFeatures;

   public HashingEncoder(int numberOfFeatures, boolean isBinary) {
      super(false);
      this.numberOfFeatures = numberOfFeatures;
      this.isBinary = isBinary;
   }

   @Override
   public Set<String> alphabet() {
      return Collections.emptySet();
   }

   @Override
   public String decode(double value) {
      return null;
   }

   @Override
   public double encode(String value) {
      return (value.hashCode() & 0x7fffffff) % numberOfFeatures;
   }

   @Override
   public void fit(Dataset dataset) {

   }

   @Override
   public int size() {
      return numberOfFeatures;
   }

   public JsonEntry toJson() {
      return JsonEntry.object()
                      .addProperty("class", HashingEncoder.class)
                      .addProperty("numberOfFeatures", numberOfFeatures)
                      .addProperty("isBinary", isBinary);
   }

   @Override
   public String toString() {
      return "HashingEncoder{" +
                "numberOfFeatures=" + numberOfFeatures +
                ", isBinary=" + isBinary +
                '}';
   }
}//END OF HashingEncoder
