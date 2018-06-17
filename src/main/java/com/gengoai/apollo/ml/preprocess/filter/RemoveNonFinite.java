package com.gengoai.apollo.ml.preprocess.filter;

import com.gengoai.apollo.ml.Instance;
import com.gengoai.apollo.ml.data.Dataset;
import com.gengoai.apollo.ml.preprocess.InstancePreprocessor;

import java.util.stream.Collectors;

/**
 * A filter implementation that removes features with non-finite or Nan values
 *
 * @author David B. Bracewell
 */
public class RemoveNonFinite implements FilterProcessor<Instance>, InstancePreprocessor {
   private static final long serialVersionUID = 1L;

   @Override
   public Instance apply(Instance example) {
      return Instance.create(example.stream()
                                    .filter(i -> !Double.isNaN(i.getValue()) && Double.isFinite(i.getValue()))
                                    .collect(Collectors.toList()), example.getLabel());
   }

   @Override
   public String describe() {
      return "RemoveNonFinite";
   }

   @Override
   public void fit(Dataset<Instance> dataset) {

   }

   @Override
   public void reset() {

   }

}// END OF RemoveNonFinite
