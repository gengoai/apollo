package com.davidbracewell.apollo.ml.sequence;

import com.davidbracewell.apollo.ml.Feature;
import com.davidbracewell.apollo.ml.Featurizer;
import com.davidbracewell.apollo.ml.Instance;
import com.davidbracewell.cache.CacheProxy;
import com.davidbracewell.cache.Cached;
import com.davidbracewell.conversion.Cast;
import com.davidbracewell.function.SerializableFunction;
import com.davidbracewell.stream.MStream;
import lombok.NonNull;

import java.util.*;

/**
 * The interface Sequence featurizer.
 *
 * @param <INPUT> the type parameter
 * @author David B. Bracewell
 */
public interface SequenceFeaturizer<INPUT> extends Featurizer<ContextualIterator<INPUT>> {

  /**
   * Feature set set.
   *
   * @param features the features
   * @return the set
   */
  static Set<Feature> featureSet(Feature... features) {
    return new HashSet<>(Arrays.asList(features));
  }

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
      @Cached
      public Set<Feature> apply(ContextualIterator<T> tContextualIterator) {
        return new HashSet<>(function.apply(tContextualIterator));
      }
    };
  }

  @Override
  default SequenceFeaturizer<INPUT> cache(String cacheName) {
    return CacheProxy.cache(this, cacheName);
  }

  @Override
  default SequenceFeaturizer<INPUT> cache() {
    return CacheProxy.cache(this);
  }


  /**
   * Chain sequence featurizer.
   *
   * @param <T>         the type parameter
   * @param featurizers the featurizers
   * @return the sequence featurizer
   */
  @SafeVarargs
  static <T> SequenceFeaturizer<T> chain(@NonNull SequenceFeaturizer<? super T>... featurizers) {
    return new SequenceFeaturizer<T>() {
      private static final long serialVersionUID = 1L;
      final Set<SequenceFeaturizer<? super T>> extractors = new LinkedHashSet<>(Arrays.asList(featurizers));

      @Override
      @Cached
      public Set<Feature> apply(ContextualIterator<T> tContextualIterator) {
        Set<Feature> features = new HashSet<>();
        extractors.forEach(ex -> features.addAll(ex.apply(Cast.as(tContextualIterator))));
        return features;
      }
    };
  }

  /**
   * Extract sequence m stream.
   *
   * @param stream the stream
   * @return the m stream
   */
  default MStream<Sequence> extractSequence(@NonNull MStream<ContextualIterator<INPUT>> stream) {
    return stream.map(this::extractSequence);
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
