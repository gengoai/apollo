package com.gengoai.apollo.ml;

import com.gengoai.Copyable;
import com.gengoai.apollo.ml.preprocess.Preprocessor;
import com.gengoai.function.SerializableFunction;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;

/**
 * @author David B. Bracewell
 */
public class PreprocessorList extends ArrayList<Preprocessor> implements SerializableFunction<Example, Example>, Copyable<PreprocessorList> {
   private static final long serialVersionUID = 1L;


   public PreprocessorList(Collection<Preprocessor> preprocessors) {
      super(preprocessors);
   }

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


   @Override
   public PreprocessorList copy() {
      PreprocessorList copy = new PreprocessorList();
      forEach(p -> copy.add(p.copy()));
      return copy;
   }
}//END OF PreprocessorList
