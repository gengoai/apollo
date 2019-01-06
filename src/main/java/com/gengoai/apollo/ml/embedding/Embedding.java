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

package com.gengoai.apollo.ml.embedding;

import com.gengoai.apollo.linear.store.VectorStore;
import com.gengoai.apollo.ml.Example;
import com.gengoai.apollo.ml.Model;
import com.gengoai.apollo.ml.ModelParameters;
import com.gengoai.apollo.ml.preprocess.Preprocessor;
import com.gengoai.collection.Streams;
import com.gengoai.string.Strings;

import java.util.List;
import java.util.stream.Collectors;

/**
 * <p>
 * An Embedding model learns a mapping between words and n-dimensional arrays. Common implementations are Word2Vec, SVD,
 * and Glove.
 * </p>
 *
 * @author David B. Bracewell
 */
public abstract class Embedding extends Model<VectorStore> {
   private static final long serialVersionUID = 1L;

   /**
    * Instantiates a new Embedding.
    *
    * @param preprocessors the preprocessors
    */
   public Embedding(Preprocessor... preprocessors) {
      super(ModelParameters.indexedLabelVectorizer().preprocessors(preprocessors));
   }

   /**
    * Instantiates a new Embedding.
    *
    * @param modelParameters the model parameters
    */
   public Embedding(ModelParameters modelParameters) {
      super(modelParameters);
   }

   /**
    * Converts an example into an ordered list of strings
    *
    * @param example the example
    * @return the list of string representing the features in the example
    */
   protected List<String> exampleToList(Example example, boolean fullFeatureName) {
      if (example.isInstance()) {
         return example.getFeatures()
                       .stream()
                       .map(f -> {
                          if (fullFeatureName) {
                             return f.name;
                          } else {
                             return f.getSuffix();
                          }
                       })
                       .filter(Strings::isNotNullOrBlank)
                       .collect(Collectors.toList());
      }
      return Streams.asStream(example)
                    .flatMap(e -> e.getFeatures().stream()
                                   .map(f -> {
                                      if (fullFeatureName) {
                                         return f.name;
                                      } else {
                                         return f.getSuffix();
                                      }
                                   }))
                    .filter(Strings::isNotNullOrBlank)
                    .collect(Collectors.toList());
   }

}//END OF Embedding
