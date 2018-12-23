package com.gengoai.apollo.ml;

import com.gengoai.conversion.Cast;

import java.io.Serializable;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

/**
 * Transforms an input into features to create examples to be trained or estimated using {@link Model}s
 *
 * @param <INPUT> the type parameter
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
   static <T> Featurizer<T> chain(Featurizer<? super T> featurizerOne, Featurizer<? super T>... featurizers) {
      if (featurizers.length == 0) {
         return Cast.as(featurizerOne);
      }
      return new FeaturizerChain<>(featurizerOne, Arrays.asList(featurizers));
   }

   /**
    * Chains this featurizer with another.
    *
    * @param featurizer the next featurizer to call
    * @return the new chain of featurizer
    */
   default Featurizer<INPUT> and(Featurizer<? super INPUT> featurizer) {
      if (this instanceof FeaturizerChain) {
         Cast.<FeaturizerChain<INPUT>>as(this).addFeaturizer(featurizer);
         return this;
      }
      return new FeaturizerChain<>(this, Collections.singleton(featurizer));
   }

   /**
    * Applies this featurizer to the given input
    *
    * @param input the input to featurize
    * @return the set of features
    */
   List<Feature> apply(INPUT input);

   /**
    * Converts the given input into features and creates an <code>Instance</code> from the features.
    *
    * @param object the input
    * @return the instance
    */
   default Example extractInstance(INPUT object) {
      return new Instance(null, apply(object));
   }

   /**
    * Converts the given input into features and creates an <code>Instance</code> from the features.
    *
    * @param object the input
    * @param label  the label to assign the input
    * @return the instance
    */
   default Example extractInstance(INPUT object, Object label) {
      return new Instance(label, apply(object));
   }

   /**
    * Converts the given input into features and creates an <code>Instance</code> from the features.
    *
    * @param labeledDatum the labeled datum to featurize
    * @return the instance
    */
   default Example extractInstance(LabeledDatum<? extends INPUT> labeledDatum) {
      return new Instance(labeledDatum.label, apply(labeledDatum.data));
   }


}//END OF Featurizer
