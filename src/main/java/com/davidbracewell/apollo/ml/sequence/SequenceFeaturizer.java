package com.davidbracewell.apollo.ml.sequence;

import com.davidbracewell.apollo.ml.Feature;
import com.davidbracewell.apollo.ml.Instance;
import com.davidbracewell.apollo.ml.featurizer.Featurizer;
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
public interface SequenceFeaturizer<INPUT> extends Featurizer<Context<INPUT>> {
   long serialVersionUID = 1L;

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
         public List<Feature> apply(Context<T> tContext) {
            List<Feature> features = new ArrayList<>();
            extractors.forEach(ex -> features.addAll(ex.apply(Cast.as(tContext))));
            return features;
         }
      };
   }

   /**
    * Of sequence featurizer.
    *
    * @param <T>      the type parameter
    * @param function the function
    * @return the sequence featurizer
    */
   static <T> SequenceFeaturizer<T> of(@NonNull SerializableFunction<Context<T>, ? extends Collection<Feature>> function) {
      return new SequenceFeaturizer<T>() {
         private static final long serialVersionUID = 1L;

         @Override
         @Cached
         public List<Feature> apply(Context<T> tContext) {
            return new ArrayList<>(function.apply(tContext));
         }
      };
   }

   @Override
   default SequenceFeaturizer<Context<INPUT>> asSequenceFeaturizer() {
      return Cast.as(this);
   }

   @Override
   default SequenceFeaturizer<INPUT> cache() {
      return CacheProxy.cache(this);
   }

   @Override
   default SequenceFeaturizer<INPUT> cache(String cacheName) {
      return CacheProxy.cache(this, cacheName);
   }

   /**
    * Extract sequence sequence.
    *
    * @param iterator the iterator
    * @return the sequence
    */
   default Sequence extractSequence(@NonNull Context<? extends INPUT> iterator) {
      ArrayList<Instance> instances = new ArrayList<>();
      while (iterator.hasNext()) {
         iterator.next();
         instances.add(extractInstance(Cast.as(iterator), iterator.getLabel()));
      }
      instances.trimToSize();
      return new Sequence(instances);
   }

   /**
    * Extract sequence m stream.
    *
    * @param stream the stream
    * @return the m stream
    */
   default MStream<Sequence> extractSequence(@NonNull MStream<Context<? extends INPUT>> stream) {
      return stream.map(this::extractSequence);
   }

}// END OF SequenceFeaturizer
