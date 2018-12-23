package com.gengoai.apollo.ml.preprocess;

import com.gengoai.apollo.ml.Instance;
import com.gengoai.apollo.ml.data.Dataset;

import java.io.Serializable;
import java.util.Optional;

/**
 * <p>Removes features whose values is non-finite.</p>
 *
 * @author David B. Bracewell
 */
public class RemoveNonFinite implements InstancePreprocessor, Serializable {
   private static final long serialVersionUID = 1L;

   @Override
   public Instance applyInstance(Instance example) {
      return example.mapFeatures(f -> Optional.ofNullable(Double.isFinite(f.value)
                                                          ? f
                                                          : null));
   }

   @Override
   public Dataset fitAndTransform(Dataset dataset) {
      return dataset.mapSelf(this::apply);
   }


   @Override
   public void reset() {

   }


   @Override
   public String toString() {
      return "RemoveNonFinite{}";
   }

}//END OF RemoveNonFinite
