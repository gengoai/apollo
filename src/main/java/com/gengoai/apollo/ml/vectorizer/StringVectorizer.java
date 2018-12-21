package com.gengoai.apollo.ml.vectorizer;

import com.gengoai.apollo.linear.NDArray;
import com.gengoai.apollo.linear.NDArrayFactory;
import com.gengoai.apollo.ml.Example;

/**
 * @author David B. Bracewell
 */
public abstract class StringVectorizer implements Vectorizer<String> {
   private final boolean isLabelVectorizer;

   protected StringVectorizer(boolean isLabelVectorizer) {
      this.isLabelVectorizer = isLabelVectorizer;
   }

   private void setIf(NDArray ndArray, String name, double value) {
      int index = (int) encode(name);
      if (index >= 0) {
         ndArray.set(index, value);
      }
   }

   private void setIf(NDArray ndArray, int row, String name, double value) {
      int index = (int) encode(name);
      if (index >= 0) {
         ndArray.set(row, index, value);
      }
   }

   @Override
   public NDArray transform(Example example) {
      if (example.isInstance()) {
         return transformInstance(example);
      }
      return transformSequence(example);
   }

   private NDArray transformInstance(Example example) {
      final NDArray ndArray = NDArrayFactory.DEFAULT().zeros(size());
      if (isLabelVectorizer) {
         example.getLabelAsSet().forEach(label -> setIf(ndArray, label, 1.0));
      } else {
         example.getFeatures().forEach(feature -> setIf(ndArray, feature.name, feature.value));
      }
      return ndArray;
   }

   private NDArray transformSequence(Example example) {
      NDArray ndArray = NDArrayFactory.DEFAULT().zeros(example.size(), size());
      for (int row = 0; row < example.size(); row++) {
         Example child = example.getExample(row);
         final int childIndex = row;
         if (isLabelVectorizer) {
            child.getLabelAsSet().forEach(label -> setIf(ndArray, childIndex, label, 1.0));
         } else {
            child.getFeatures().forEach(feature -> setIf(ndArray, childIndex, feature.name, feature.value));
         }
      }
      return ndArray;
   }



}//END OF StringVectorizer
