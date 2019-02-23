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

import com.gengoai.apollo.linear.NDArray;
import com.gengoai.apollo.linear.NDArrayFactory;
import com.gengoai.apollo.linear.VectorComposition;
import com.gengoai.apollo.ml.DiscreteModel;
import com.gengoai.apollo.ml.DiscretePipeline;
import com.gengoai.apollo.ml.Example;
import com.gengoai.apollo.ml.FitParameters;
import com.gengoai.apollo.ml.data.Dataset;
import com.gengoai.apollo.ml.preprocess.Preprocessor;
import com.gengoai.collection.Streams;
import com.gengoai.conversion.Cast;
import com.gengoai.io.resource.Resource;
import com.gengoai.string.Strings;

import javax.ws.rs.NotSupportedException;
import java.io.BufferedReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/**
 * @author David B. Bracewell
 */
public class Embedding extends DiscreteModel {
   private static final long serialVersionUID = 1L;
   protected VectorIndex vectorIndex;

   protected Embedding(Preprocessor... preprocessors) {
      super(preprocessors);
   }

   protected Embedding(DiscretePipeline modelParameters) {
      super(modelParameters);
   }


   /**
    * Creates a vector using the given vector composition for the given words.
    *
    * @param composition the composition function to use
    * @param words       the words whose vectors we want to compose
    * @return a composite vector consisting of the given words and calculated using the given vector composition
    */
   public NDArray compose(VectorComposition composition, String... words) {
      if (words == null) {
         return NDArrayFactory.DEFAULT().zeros(dimension());
      } else if (words.length == 1) {
         return lookup(words[0]);
      }
      return composition.compose(Arrays.stream(words)
                                       .map(this::lookup)
                                       .collect(Collectors.toList()));
   }

   /**
    * Gets alphabet.
    *
    * @return the alphabet
    */
   public Set<String> getAlphabet() {
      return getPipeline().featureVectorizer.alphabet();
   }

   /**
    * Lookup nd array.
    *
    * @param key the key
    * @return the nd array
    */
   public NDArray lookup(String key) {
      int index = getPipeline().featureVectorizer.indexOf(key);
      if (index < 0) {
         return NDArrayFactory.DENSE.zeros(dimension());
      }
      return vectorIndex.lookup(index);
   }

   /**
    * Queries the vector store to find similar vectors to the given {@link VSQuery}.
    *
    * @param query the query to use find similar vectors
    * @return Stream of vectors matching the query
    */
   public Stream<NDArray> query(VSQuery query) {
      NDArray queryVector = query.queryVector(this);
      return query.applyFilters(vectorIndex.stream()
                                           .parallel()
                                           .map(v -> v.copy().setWeight(query.measure().calculate(v, queryVector))));
   }

   /**
    * Size int.
    *
    * @return the int
    */
   public int size() {
      return getPipeline().featureVectorizer.size();
   }


   @Override
   protected void fitPreprocessed(Dataset preprocessed, FitParameters fitParameters) {
      throw new NotSupportedException();
   }

   @Override
   public FitParameters<?> getDefaultFitParameters() {
      return new FitParameters<>();
   }


   /**
    * The dimension of the vectors in the store
    *
    * @return the dimension of the vectors
    */
   public int dimension() {
      return vectorIndex.dimension();
   }


   public static Embedding readApolloBinaryFormat(Resource resource) throws IOException {
      try {
         return Cast.as(resource.readObject(), Embedding.class);
      } catch (Exception e) {
         throw new IOException(e);
      }
   }

   /**
    * Read embedding.
    *
    * @param resource the resource
    * @return the embedding
    * @throws IOException the io exception
    */
   public static Embedding readWord2VecTextFormat(Resource resource) throws IOException {
      return readWord2VecTextFormat(resource, VSTextUtils.determineUnknownWord(resource));
   }

   /**
    * Read embedding.
    *
    * @param resource    the resource
    * @param unknownWord the unknown word
    * @return the embedding
    * @throws IOException the io exception
    */
   public static Embedding readWord2VecTextFormat(Resource resource, String unknownWord) throws IOException {
      int dimension = VSTextUtils.determineDimension(resource);
      List<NDArray> ndArrays = new ArrayList<>();
      List<String> keys = new ArrayList<>();
      try (BufferedReader reader = new BufferedReader(resource.reader())) {
         String line;
         while ((line = reader.readLine()) != null) {
            if (Strings.isNotNullOrBlank(line) && !line.startsWith("#")) {
               NDArray v = VSTextUtils.convertLineToVector(line, dimension);
               ndArrays.add(v);
               if (v.scalarNorm2() == 0) {
                  System.out.println(line);
               }
               keys.add(v.getLabel());
            }
         }
      }
//      return new Embedding(keys, ndArrays.toArray(new NDArray[0]));
      return null;
   }

   /**
    * Converts an example into an ordered list of strings
    *
    * @param example         the example
    * @param fullFeatureName the full feature name
    * @return the list of string representing the features in the example
    */
   protected List<String> exampleToList(Example example, boolean fullFeatureName) {
      if (example.isInstance()) {
         return example.getFeatures()
                       .stream()
                       .map(f -> {
                          if (fullFeatureName) {
                             return f.getName();
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
                                         return f.getName();
                                      } else {
                                         return f.getSuffix();
                                      }
                                   }))
                    .filter(Strings::isNotNullOrBlank)
                    .collect(Collectors.toList());
   }

}//END OF Embedding
