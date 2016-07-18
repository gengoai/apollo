package com.davidbracewell.apollo.ml.preprocess;

import com.davidbracewell.apollo.ml.Instance;
import com.davidbracewell.apollo.ml.sequence.Sequence;

/**
 * The interface Instance preprocessor.
 *
 * @author David B. Bracewell
 */
public interface InstancePreprocessor extends Preprocessor<Instance> {

  /**
   * As sequence processor preprocessor.
   *
   * @return the preprocessor
   */
  default Preprocessor<Sequence> asSequenceProcessor() {
    return new SequencePreprocessor(this);
  }

}// END OF InstancePreprocessor
