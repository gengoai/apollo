package com.gengoai.apollo.ml.featurizer;

import com.gengoai.apollo.ml.Feature;
import com.gengoai.apollo.ml.Instance;
import com.gengoai.apollo.ml.LabeledDatum;
import com.gengoai.apollo.ml.sequence.SequenceFeaturizer;
import com.gengoai.conversion.Cast;
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
   List<Feature> apply(INPUT input);


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
      return Instance.create(apply(labeledDatum.data), labeledDatum.label);
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
