package com.davidbracewell.apollo.ml.sequence;

import java.io.Serializable;

/**
 * @author David B. Bracewell
 */
public interface SequenceValidator extends Serializable {

  SequenceValidator ALWAYS_TRUE = new SequenceValidator() {
    private static final long serialVersionUID = 1L;

    @Override
    public boolean isValid(String label, String previousLabel) {
      return true;
    }
  };

  boolean isValid(String label, String previousLabel);

}// END OF SequenceValidator
