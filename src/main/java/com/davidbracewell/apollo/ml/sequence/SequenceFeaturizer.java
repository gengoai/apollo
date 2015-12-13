package com.davidbracewell.apollo.ml.sequence;

import com.davidbracewell.apollo.ml.Feature;
import com.davidbracewell.apollo.ml.Featurizer;
import com.davidbracewell.apollo.ml.Instance;
import com.davidbracewell.function.SerializableFunction;
import lombok.NonNull;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;

/**
 * @author David B. Bracewell
 */
public interface SequenceFeaturizer<INPUT> extends Featurizer<ContextualIterator<INPUT>> {


  static <T> SequenceFeaturizer<T> of(@NonNull SerializableFunction<ContextualIterator<T>, List<Feature>> function) {
    return (SequenceFeaturizer<T>) tContextualIterator -> new HashSet<>(function.apply(tContextualIterator));
  }

  default Sequence extractSequence(@NonNull ContextualIterator<INPUT> iterator) {
    ArrayList<Instance> instances = new ArrayList<>();
    while (iterator.hasNext()) {
      iterator.next();
      instances.add(extract(iterator, iterator.getLabel()));
    }
    instances.trimToSize();
    return new Sequence(instances);
  }


}// END OF SequenceFeaturizer
