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

import com.gengoai.apollo.linear.NDArray;
import com.gengoai.apollo.linear.NDArrayFactory;
import com.gengoai.apollo.linear.VectorComposition;
import com.gengoai.apollo.linear.VectorCompositions;
import com.gengoai.apollo.linear.store.VectorStore;
import com.gengoai.apollo.ml.Example;
import com.gengoai.apollo.ml.Feature;
import com.gengoai.apollo.ml.data.Dataset;
import com.gengoai.collection.Lists;
import com.gengoai.string.Strings;

import java.util.ArrayList;
import java.util.List;
import java.util.Set;

/**
 * The type Embedding.
 *
 * @author David B. Bracewell
 */
public class Embedding extends StringVectorizer {
   private static final long serialVersionUID = 1L;
   private VectorComposition composition;
   private String unknownWord;
   private VectorStore vectorStore;


   /**
    * Instantiates a new Embedding.
    *
    * @param vectorStore the vector store
    */
   public Embedding(VectorStore vectorStore) {
      this(vectorStore, null, VectorCompositions.Average);
   }

   /**
    * Instantiates a new Embedding.
    *
    * @param vectorStore the vector store
    * @param unknownWord the unknown word
    * @param composition the composition
    */
   public Embedding(VectorStore vectorStore, String unknownWord, VectorComposition composition) {
      super(false);
      this.vectorStore = vectorStore;
      this.unknownWord = Strings.blankToNull(unknownWord);
      this.composition = composition;
   }

   @Override
   public Set<String> alphabet() {
      return vectorStore.keySet();
   }

   @Override
   public String decode(double value) {
      throw new UnsupportedOperationException();
   }

   @Override
   public double encode(String value) {
      throw new UnsupportedOperationException();
   }

   @Override
   public void fit(Dataset dataset) {

   }

   @Override
   public int size() {
      return vectorStore.size();
   }

   @Override
   protected NDArray transformInstance(Example example) {
      List<NDArray> arrays = new ArrayList<>();
      for (Feature feature : example.getFeatures()) {
         String name = feature.getSuffix();
         if (vectorStore.containsKey(name)) {
            arrays.add(vectorStore.get(name).mul(feature.value));
         } else if (unknownWord != null && vectorStore.containsKey(unknownWord)) {
            arrays.add(vectorStore.get(unknownWord).mul(feature.value));
         } else {
            arrays.add(NDArrayFactory.DEFAULT().zeros(vectorStore.dimension()));
         }
      }
      return composition.compose(arrays);
   }

   @Override
   protected NDArray transformSequence(Example example) {
      List<NDArray> rows = new ArrayList<>();
      int maxC = 0;
      for (Example instance : example) {
         NDArray n = transformInstance(instance);
         if (n.isColumnVector()) {
            n = n.T();
         }
         rows.add(n);
         maxC = Math.max(maxC, n.numCols());
      }
      final int dim = maxC;
      return NDArrayFactory.DEFAULT().vstack(
         Lists.transform(rows, n -> {
            if (n.numCols() == dim) {
               return n;
            }
            NDArray nn = NDArrayFactory.DEFAULT().zeros(1, dim);
            n.sparseIterator().forEachRemaining(e -> nn.set(e.getRow(), e.getColumn(), e.getValue()));
            return nn;
         }));
   }
}//END OF Embedding
