package com.davidbracewell.apollo.ml.featurizer;

import com.davidbracewell.apollo.ml.Feature;
import com.davidbracewell.apollo.ml.Instance;
import com.davidbracewell.apollo.ml.LabeledDatum;
import com.davidbracewell.apollo.ml.sequence.SequenceFeaturizer;
import com.davidbracewell.cache.CacheProxy;
import com.davidbracewell.cache.Cached;
import com.davidbracewell.conversion.Cast;
import lombok.NonNull;

import java.io.Serializable;
import java.util.List;

/**
 * @author David B. Bracewell
 */
@FunctionalInterface
public interface Featurizer<INPUT> extends Serializable {

   /**
    * Chains multiple featurizers together with each being called on the input data.
    *
    * @param <T>           the example type parameter
    * @param featurizerOne the first featurizer
    * @param featurizers   the featurizers to chain together
    * @return the Chained featurizers
    */
   @SafeVarargs
   static <T> Featurizer<T> chain(@NonNull Featurizer<? super T> featurizerOne, Featurizer<? super T>... featurizers) {
      if (featurizers.length == 0) {
         return Cast.as(featurizerOne);
      }
      return new FeaturizerChain<>(featurizerOne, featurizers);
   }

   /**
    * Chains this featurizer with another.
    *
    * @param featurizer the next featurizer to call
    * @return the new chain of featurizer
    */
   default Featurizer<INPUT> and(@NonNull Featurizer<? super INPUT> featurizer) {
      if (this instanceof FeaturizerChain) {
         Cast.<FeaturizerChain<INPUT>>as(this).addFeaturizer(featurizer);
         return this;
      }
      return new FeaturizerChain<>(this, featurizer);
   }

   /**
    * Applies this featurizer to the given input
    *
    * @param input the input to featurize
    * @return the set of features
    */
   @Cached
   List<Feature> apply(INPUT input);

   /**
    * Cache featurizer.
    *
    * @param cacheName the cache name
    * @return the featurizer
    */
   default Featurizer<INPUT> cache(String cacheName) {
      return CacheProxy.cache(this, cacheName);
   }

   /**
    * Caches the call to featurizer.
    *
    * @return the featurizer
    */
   default Featurizer<INPUT> cache() {
      return CacheProxy.cache(this);
   }

   /**
    * Converts the given input into features and creates an <code>Instance</code> from the features.
    *
    * @param object the input
    * @return the instance
    */
   default Instance extractInstance(@NonNull INPUT object) {
      return Instance.create(apply(object));
   }

   /**
    * Converts the given input into features and creates an <code>Instance</code> from the features.
    *
    * @param object the input
    * @param label  the label to assign the input
    * @return the instance
    */
   default Instance extractInstance(@NonNull INPUT object, Object label) {
      return Instance.create(apply(object), label);
   }

   /**
    * Converts the given input into features and creates an <code>Instance</code> from the features.
    *
    * @param labeledDatum the labeled datum to featurize
    * @return the instance
    */
   default Instance extractInstance(@NonNull LabeledDatum<? extends INPUT> labeledDatum) {
      return Instance.create(apply(labeledDatum.getData()), labeledDatum.getLabel());
   }


   /**
    * Converts this instance featurizer into a <code>SequenceFeaturizer</code> that acts on the current item in the
    * sequence.
    *
    * @return the sequence featurizer
    */
   default SequenceFeaturizer<INPUT> asSequenceFeaturizer() {
      return itr -> apply(itr.getCurrent());
   }

}// END OF Featurizer
