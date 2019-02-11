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

package com.gengoai.apollo.ml.vectorizer;

import com.gengoai.Validation;
import com.gengoai.apollo.linear.NDArray;
import com.gengoai.apollo.linear.NDArrayFactory;
import com.gengoai.apollo.linear.store.VectorStore;
import com.gengoai.apollo.ml.Example;
import com.gengoai.apollo.ml.Feature;
import com.gengoai.apollo.ml.data.Dataset;
import com.gengoai.collection.Index;
import com.gengoai.collection.Indexes;
import com.gengoai.string.Strings;

import java.util.List;
import java.util.Set;

/**
 * <p>A vectorizer that maps a single feature in an instance into a {@link NDArray} using pre-defined embeddings stored
 * in a {@link VectorStore}</p>
 *
 * @author David B. Bracewell
 */
public class EmbeddingVectorizer implements DiscreteVectorizer {
   private static final long serialVersionUID = 1L;
   private String unknownWord;
   private VectorStore vectorStore;
   private Index<String> alphabetIndex;

   /**
    * Instantiates a new Embedding.
    *
    * @param vectorStore the vector store
    */
   public EmbeddingVectorizer(VectorStore vectorStore) {
      this(vectorStore, null);
   }

   /**
    * Instantiates a new Embedding.
    *
    * @param vectorStore the vector store
    * @param unknownWord the unknown word
    */
   public EmbeddingVectorizer(VectorStore vectorStore, String unknownWord) {
      this.vectorStore = vectorStore;
      this.unknownWord = Strings.blankToNull(unknownWord);
      this.alphabetIndex = Indexes.indexOf(vectorStore.keySet().stream().sorted());
   }

   @Override
   public Set<String> alphabet() {
      return vectorStore.keySet();
   }

   @Override
   public String getString(double value) {
      if (value < 0 || value >= alphabetIndex.size()) {
         return unknownWord;
      }
      return alphabetIndex.get((int) value);
   }

   @Override
   public int indexOf(String value) {
      if (alphabetIndex.contains(value)) {
         return alphabetIndex.getId(value);
      }
      return unknownWord == null ? -1 : alphabetIndex.getId(unknownWord);
   }


   @Override
   public void fit(Dataset dataset) {

   }

   @Override
   public NDArray transform(Example example) {
      List<Feature> features = example.getFeatures();
      Validation.checkArgument(features.size() == 1, "Only supports one feature per instance.");
      Feature feature = features.get(0);
      String name = feature.getSuffix();
      if (vectorStore.containsKey(name)) {
         return vectorStore.get(name).mul(feature.getValue());
      } else if (unknownWord != null && vectorStore.containsKey(unknownWord)) {
         return vectorStore.get(unknownWord).mul(feature.getValue());
      } else {
         return NDArrayFactory.DEFAULT().zeros(vectorStore.dimension());
      }
   }

   @Override
   public int size() {
      return vectorStore.size();
   }

}//END OF Embedding
