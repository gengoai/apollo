package com.davidbracewell.apollo.ml.sequence;

import com.davidbracewell.apollo.ml.Feature;
import com.davidbracewell.apollo.ml.Featurizer;
import com.davidbracewell.apollo.ml.Instance;
import com.davidbracewell.function.SerializableFunction;
import lombok.NonNull;

import java.util.ArrayList;
import java.util.Collection;
import java.util.HashSet;
import java.util.Set;

/**
 * The interface Sequence featurizer.
 *
 * @param <INPUT> the type parameter
 * @author David B. Bracewell
 */
public interface SequenceFeaturizer<INPUT> extends Featurizer<ContextualIterator<INPUT>> {


  /**
   * Of sequence featurizer.
   *
   * @param <T>      the type parameter
   * @param function the function
   * @return the sequence featurizer
   */
  static <T> SequenceFeaturizer<T> of(@NonNull SerializableFunction<ContextualIterator<T>, ? extends Collection<Feature>> function) {
    return new SequenceFeaturizer<T>() {
      private static final long serialVersionUID = 1L;

      @Override
      public Set<Feature> apply(ContextualIterator<T> tContextualIterator) {
        return new HashSet<>(function.apply(tContextualIterator));
      }
    };
  }

  /**
   * Extract sequence sequence.
   *
   * @param iterator the iterator
   * @return the sequence
   */
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
