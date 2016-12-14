/*
 * (c) 2005 David B. Bracewell
 *
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package com.davidbracewell.apollo.ml;

import com.davidbracewell.apollo.ml.sequence.SequenceFeaturizer;
import com.davidbracewell.cache.CacheProxy;
import com.davidbracewell.cache.Cached;
import com.davidbracewell.conversion.Cast;
import lombok.NonNull;

import java.io.Serializable;
import java.util.Set;

/**
 * <p> A featurizer converts an input into a one or more <code>Feature</code>s which have a name and a value. Specific
 * implementations may implement the generic featurizer interface or specialize the Binary or Real featurizers. </p>
 *
 * @param <INPUT> the type of the input being converted into features
 * @author David B. Bracewell
 */
@FunctionalInterface
public interface Featurizer<INPUT> extends Serializable {
   long serialVersionUID = 1L;

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
    * Applies this featurizer to the given input
    *
    * @param input the input to featurize
    * @return the set of features
    */
   @Cached
   Set<Feature> apply(INPUT input);


   /**
    * Caches the call to featurizer.
    *
    * @return the featurizer
    */
   default Featurizer<INPUT> cache() {
      return CacheProxy.cache(this);
   }

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
   default Instance extractInstance(@NonNull LabeledDatum<INPUT> labeledDatum) {
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

}//END OF Featurizer
