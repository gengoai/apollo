package com.gengoai.apollo.ml.vectorizer;

import com.gengoai.apollo.linear.NDArray;
import com.gengoai.apollo.linear.NDArrayFactory;
import com.gengoai.apollo.ml.Example;
import com.gengoai.apollo.ml.Feature;

import java.util.List;

/**
 * @author David B. Bracewell
 */
public abstract class StringVectorizer implements Vectorizer<String> {

   @Override
   public NDArray transform(Example example) {
      NDArray ndArray = NDArrayFactory.DEFAULT().zeros(example.size(), size());
      for (int row = 0; row < example.size(); row++) {
         Example child = example.getExample(row);
         if (isLabelVectorizer()) {
            for (String label : child.getLabelAsSet()) {
               int index = (int) encode(label);
               if (index >= 0) {
                  ndArray.set(row, index, 1.0);
               }
            }
         } else {
            List<Feature> features = child.getFeatures();
            for (Feature feature : features) {
               int index = (int) encode(feature.name);
               if (index >= 0) {
                  ndArray.set(row, index, feature.value);
               }
            }
         }
      }
      return ndArray;
   }

}//END OF StringVectorizer
