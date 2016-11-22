package com.davidbracewell.apollo.ml.preprocess;

import com.davidbracewell.apollo.ml.Instance;

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
