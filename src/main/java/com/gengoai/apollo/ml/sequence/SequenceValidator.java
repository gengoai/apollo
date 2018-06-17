package com.gengoai.apollo.ml.sequence;

import com.gengoai.apollo.ml.Instance;

import java.io.Serializable;

/**
 * @author David B. Bracewell
 */
public interface SequenceValidator extends Serializable {

   SequenceValidator ALWAYS_TRUE = new SequenceValidator() {
      private static final long serialVersionUID = 1L;

      @Override
      public boolean isValid(String label, String previousLabel, Instance instance) {
         return true;
      }
   };

   boolean isValid(String label, String previousLabel, Instance instance);

}// END OF SequenceValidator
