package com.gengoai.apollo.ml.vectorizer;

import com.gengoai.apollo.linear.NDArray;
import com.gengoai.apollo.ml.Example;
import com.gengoai.apollo.ml.data.Dataset;

/**
 * @author David B. Bracewell
 */
public class NoOptVectorizer<T> implements Vectorizer<T> {

   @Override
   public T decode(double value) {
      return null;
   }

   @Override
   public double encode(T value) {
      return 0;
   }

   @Override
   public void fit(Dataset dataset) {

   }

   @Override
   public int size() {
      return 0;
   }

   @Override
   public NDArray transform(Example example) {
      return null;
   }
}//END OF NoOptVectorizer
