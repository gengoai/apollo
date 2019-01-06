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
 *
 */

package com.gengoai.apollo.ml;

import java.util.List;

/**
 * <p>
 * A featurizer extractor converts an input object into an {@link Example}. Specializations of this class are {@link
 * Featurizer} that extract features based on a single object and {@link ContextFeaturizer} which extract features based
 * on the objects and its context.
 * </p>
 *
 * @param <I> the type parameter for the object being converted to an example.
 * @author David B. Bracewell
 */
public interface FeatureExtractor<I> {

   /**
    * Applies only the contextual extractors to the given sequence.
    *
    * @param sequence the sequence to generate contextual features for
    * @return the example with contextual features
    */
   default Example contextualize(Example sequence) {
      return sequence;
   }

   /**
    * Extracts an {@link Instance} example from the given input data
    *
    * @param input the datum to extract from
    * @return the example
    */
   Example extract(I input);

   /**
    * Extracts a {@link Sequence} example from the given labeled sequence
    *
    * @param sequence the sequence of labeled items to extractor from
    * @return the example
    */
   default Example extract(LabeledSequence<? extends I> sequence) {
      Sequence out = new Sequence();
      sequence.forEach(i -> out.add(extract(i)));
      return contextualize(out);
   }

   /**
    * Extracts an {@link Instance} example from the given labeled data
    *
    * @param datum the datum to extract from
    * @return the example
    */
   default Example extract(LabeledDatum<? extends I> datum) {
      Example ii = extract(datum.data);
      ii.setLabel(datum.label);
      return ii;
   }

   /**
    * Extracts a {@link Sequence} example from the given list of items
    *
    * @param sequence the sequence of labeled items to extractor from
    * @return the example
    */
   default Example extract(List<? extends I> sequence) {
      Sequence out = new Sequence();
      for (I i : sequence) {
         out.add(extract(i));
      }
      return contextualize(out);
   }


}//END OF FeatureExtractor
