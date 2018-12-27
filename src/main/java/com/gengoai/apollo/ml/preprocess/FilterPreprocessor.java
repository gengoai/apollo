package com.gengoai.apollo.ml.preprocess;

import com.gengoai.apollo.ml.Instance;
import com.gengoai.apollo.ml.data.Dataset;

import java.io.Serializable;
import java.util.Collection;
import java.util.Set;
import java.util.stream.Collectors;

import static com.gengoai.collection.Sets.asHashSet;
import static com.gengoai.collection.Sets.hashSetOf;

/**
 * The type Filter preprocessor.
 *
 * @author David B. Bracewell
 */
public class FilterPreprocessor implements InstancePreprocessor, Serializable {
   private static final long serialVersionUID = 1L;
   private final Set<String> toRemove;

   /**
    * Instantiates a new Filter preprocessor.
    *
    * @param toRemove the to remove
    */
   public FilterPreprocessor(String... toRemove) {
      this.toRemove = hashSetOf(toRemove);
   }

   /**
    * Instantiates a new Filter preprocessor.
    *
    * @param toRemove the to remove
    */
   public FilterPreprocessor(Collection<String> toRemove) {
      this.toRemove = asHashSet(toRemove);
   }

   @Override
   public Instance applyInstance(Instance example) {
      Instance ii = new Instance(example.getLabel(), example.getFeatures().stream()
                                                            .filter(f -> !toRemove.contains(f.getPrefix()))
                                                            .collect(Collectors.toList()));
      ii.setWeight(example.getWeight());
      return ii;
   }

   @Override
   public Dataset fitAndTransform(Dataset dataset) {
      return dataset.map(this::apply);
   }

   @Override
   public void reset() {

   }
}//END OF FilterPreprocessor
