package com.gengoai.apollo.ml.preprocess;

import com.gengoai.apollo.ml.Example;
import com.gengoai.apollo.ml.Instance;

/**
 * <p>A preprocessor built to work on Instance examples</p>
 *
 * @author David B. Bracewell
 */
public interface InstancePreprocessor extends Preprocessor {

   @Override
   default Example apply(Example example) {
      return example.mapInstance(this::applyInstance);
   }

   /**
    * Process instance instance.
    *
    * @param example the example
    * @return the instance
    */
   Instance applyInstance(Instance example);


}//END OF InstancePreprocessor
