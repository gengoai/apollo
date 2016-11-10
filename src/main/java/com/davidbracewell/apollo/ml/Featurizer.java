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
import com.davidbracewell.stream.MStream;
import com.google.common.base.Preconditions;
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


   /**
    * Chain featurizer.
    *
    * @param <T>        the type parameter
    * @param extractors the extractors
    * @return the featurizer
    */
   @SafeVarargs
   static <T> Featurizer<T> chain(@NonNull Featurizer<? super T>... extractors) {
      Preconditions.checkState(extractors.length > 0, "No Featurizers have been specified.");
      if (extractors.length == 1) {
         return Cast.as(extractors[0]);
      }
      return new FeaturizerChain<>(extractors);
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
    * As sequence featurizer sequence featurizer.
    *
    * @return the sequence featurizer
    */
   default SequenceFeaturizer<INPUT> asSequenceFeaturizer() {
      return itr -> apply(itr.getCurrent());
   }

   /**
    * Cache featurizer.
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
   default Featurizer<INPUT> chain(@NonNull Featurizer<? super INPUT> featurizer) {
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
   default Instance extract(@NonNull INPUT object) {
      return Instance.create(apply(object));
   }

   /**
    * Converts the given input into features and creates an <code>Instance</code> from the features.
    *
    * @param object the input
    * @param label  the label to assign the input
    * @return the instance
    */
   default Instance extract(@NonNull INPUT object, Object label) {
      return Instance.create(apply(object), label);
   }

   /**
    * Extract m stream.
    *
    * @param inputStream the input stream
    * @return the m stream
    */
   default MStream<Instance> extract(@NonNull MStream<INPUT> inputStream) {
      return inputStream.map(this::extract);
   }

   /**
    * Extract instance.
    *
    * @param labeledDatum the labeled datum
    * @return the instance
    */
   default Instance extractLabeled(@NonNull LabeledDatum<INPUT> labeledDatum) {
      return Instance.create(apply(labeledDatum.getData()), labeledDatum.getLabel());
   }

   /**
    * Extract labeled m stream.
    *
    * @param inputStream the input stream
    * @return the m stream
    */
   default MStream<Instance> extractLabeled(@NonNull MStream<LabeledDatum<INPUT>> inputStream) {
      return inputStream.map(this::extractLabeled);
   }


}//END OF Featurizer
