package com.gengoai.apollo.ml.preprocess;

import com.gengoai.apollo.ml.Instance;
import com.gengoai.apollo.ml.Instance;

/**
 * Preprocessor for instance based examples
 *
 * @author David B. Bracewell
 */
public interface InstancePreprocessor extends Preprocessor<Instance> {

   /**
    * Converts the instance preprocessor into a sequence processor
    *
    * @return the sequence preprocessor
    */
   default SequencePreprocessor asSequenceProcessor() {
      return new SequencePreprocessor(this);
   }

}// END OF InstancePreprocessor
