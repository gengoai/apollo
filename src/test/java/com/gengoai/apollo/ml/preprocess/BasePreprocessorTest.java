package com.gengoai.apollo.ml.preprocess;

import com.gengoai.apollo.ml.Example;
import com.gengoai.apollo.ml.data.Dataset;
import org.junit.Test;

import static org.junit.Assert.*;

/**
 * @author David B. Bracewell
 */
public abstract class BasePreprocessorTest {

   private final Preprocessor preprocessor;

   protected BasePreprocessorTest(Preprocessor preprocessor) {
      this.preprocessor = preprocessor;
   }


   public Dataset loadDataset() {
      return null;
   }

   @Test
   public final void fitAndTransform() {
      for (Example example : preprocessor.fitAndTransform(loadDataset())) {
         assertTrue(testTransform(example));
      }
   }


   public abstract boolean testTransform(Example example);

}