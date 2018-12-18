package com.gengoai.apollo.ml.preprocess;

import com.gengoai.apollo.ml.Example;
import com.gengoai.apollo.ml.data.Dataset;
import com.gengoai.function.SerializableFunction;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;

/**
 * <p>An array list of preprocessors that can act as a transformation over examples to preprocess them and supplies a
 * convenience method for fitting all preprocessors in the list.</p>
 *
 * @author David B. Bracewell
 */
public class PreprocessorList extends ArrayList<Preprocessor> implements SerializableFunction<Example, Example> {
   private static final long serialVersionUID = 1L;


   /**
    * Instantiates a new Preprocessor list.
    *
    * @param preprocessors the preprocessors
    */
   public PreprocessorList(Collection<Preprocessor> preprocessors) {
      super(preprocessors);
   }

   /**
    * Instantiates a new Preprocessor list.
    *
    * @param preprocessors the preprocessors
    */
   public PreprocessorList(Preprocessor... preprocessors) {
      Collections.addAll(this, preprocessors);
   }


   @Override
   public Example apply(Example example) {
      for (Preprocessor preprocessor : this) {
         example = preprocessor.apply(example);
      }
      return example;
   }

   /**
    * Performs a fit for each preprocessor in the list.
    *
    * @param dataset the dataset to fit the preprocessors over
    */
   public Dataset fitAndTransform(Dataset dataset) {
      for (Preprocessor preprocessor : this) {
         dataset = preprocessor.fitAndTransform(dataset);
      }
      return dataset;
   }


   @Override
   public String toString() {
      StringBuilder toString = new StringBuilder("Preprocessors[");
      for (int i = 0; i < size(); i++) {
         if (i > 0) {
            toString.append(", ");
         }
         toString.append(get(i));
      }
      return toString.append("]").toString();
   }

}//END OF PreprocessorList
