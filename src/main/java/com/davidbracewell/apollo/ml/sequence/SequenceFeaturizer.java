package com.davidbracewell.apollo.ml.sequence;

import com.davidbracewell.apollo.ml.Feature;
import com.davidbracewell.apollo.ml.Featurizer;
import com.davidbracewell.apollo.ml.Instance;
import com.davidbracewell.function.SerializableFunction;
import com.sun.istack.internal.NotNull;
import lombok.NonNull;

import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;

/**
 * @author David B. Bracewell
 */
public interface SequenceFeaturizer<INPUT> extends Featurizer<ContextualIterator<INPUT>> {


  static <T> SequenceFeaturizer<T> of(@NonNull SerializableFunction<ContextualIterator<T>, List<Feature>> function) {
    return (SequenceFeaturizer<T>) tContextualIterator -> new HashSet<>(function.apply(tContextualIterator));
  }

  default Sequence extractSequence(@NotNull ContextualIterator<INPUT> iterator) {
    List<Instance> instances = new LinkedList<>();
    while (iterator.hasNext()) {
      iterator.next();
      instances.add(extract(iterator, iterator.getLabel()));
    }
    return new Sequence(instances);
  }


}// END OF SequenceFeaturizer
