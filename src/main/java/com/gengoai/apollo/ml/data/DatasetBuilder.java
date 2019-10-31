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

package com.gengoai.apollo.ml.data;

import com.gengoai.apollo.ml.Example;
import com.gengoai.apollo.ml.FeatureExtractor;
import com.gengoai.apollo.ml.LabeledDatum;
import com.gengoai.apollo.ml.LabeledSequence;
import com.gengoai.apollo.ml.data.format.DataFormat;
import com.gengoai.io.resource.Resource;
import com.gengoai.stream.MStream;
import com.gengoai.stream.StreamingContext;

import java.io.IOException;
import java.util.Collection;
import java.util.List;

/**
 * <p>
 * Builder pattern for creating {@link Dataset} based on given {@link DatasetType}.
 * </p>
 *
 * @author David B. Bracewell
 */
public class DatasetBuilder {
   private DatasetType type = DatasetType.InMemory;
   private DataFormat dataFormat = null;

   /**
    * Sets the type of Dataset to be built (by default InMemory)
    *
    * @param type the Dataset type
    * @return This DatasetBuilder
    */
   public DatasetBuilder type(DatasetType type) {
      this.type = type;
      return this;
   }

   /**
    * Sets the {@link DataFormat} to use when calling {@link #source(Resource)}.
    *
    * @param dataFormat the data format
    * @return This DatasetBuilder
    */
   public DatasetBuilder dataFormat(DataFormat dataFormat) {
      this.dataFormat = dataFormat;
      return this;
   }

   /**
    * Creates a Dataset from the given input data transforming it to examples using the given {@link FeatureExtractor}
    *
    * @param <I>       the input type parameter
    * @param instances the raw input instances which will be transformed into examples using the given {@link
    *                  FeatureExtractor}
    * @param extractor the feature extractor to transform input objects to examples
    * @return the dataset
    */
   public <I> Dataset instances(MStream<I> instances, FeatureExtractor<? super I> extractor) {
      return type.create(instances.map(extractor::extractExample));
   }

   /**
    * Creates a Dataset from the given labeled data transforming it to examples using the given {@link
    * FeatureExtractor}
    *
    * @param <I>       the input type parameter
    * @param instances the labeled input instances which will be transformed into examples using the given {@link
    *                  FeatureExtractor}
    * @param extractor the extractor
    * @return the dataset
    */
   public <I> Dataset labeledInstances(MStream<LabeledDatum<I>> instances, FeatureExtractor<? super I> extractor) {
      return type.create(instances.map(extractor::extractExample));
   }

   /**
    * Creates a Dataset from the given sequence input data transforming it to examples using the given {@link
    * FeatureExtractor}
    *
    * @param <I>       the input type parameter
    * @param instances the raw input sequences which will be transformed into examples using the given {@link
    *                  FeatureExtractor}
    * @param extractor the feature extractor to transform input objects to examples
    * @return the dataset
    */
   public <I> Dataset sequences(MStream<List<? extends I>> instances, FeatureExtractor<? super I> extractor) {
      return type.create(instances.map(extractor::extractExample));
   }

   /**
    * Creates a Dataset from the given sequence input data transforming it to examples using the given {@link
    * FeatureExtractor}
    *
    * @param <I>       the input type parameter
    * @param instances the raw input sequences which will be transformed into examples using the given {@link
    *                  FeatureExtractor}
    * @param extractor the feature extractor to transform input objects to examples
    * @return the dataset
    */
   public <I> Dataset sequences(Collection<List<I>> instances, FeatureExtractor<I> extractor) {
      return type.create(StreamingContext.local().stream(instances.stream().map(extractor::extractExample)));
   }

   /**
    * Creates a Dataset from the given labeled data transforming it to examples using the given {@link
    * FeatureExtractor}
    *
    * @param <I>       the input type parameter
    * @param instances the labeled input sequences which will be transformed into examples using the given {@link
    *                  FeatureExtractor}
    * @param extractor the feature extractor to transform input objects to examples
    * @return the dataset
    */
   public <I> Dataset labeledSequences(MStream<LabeledSequence<I>> instances, FeatureExtractor<? super I> extractor) {
      return type.create(instances.map(extractor::extractExample));
   }

   /**
    * Creates a Dataset from the given stream of examples
    *
    * @param stream the stream of examples
    * @return the dataset
    */
   public Dataset source(MStream<Example> stream) {
      return type.create(stream);
   }

   /**
    * Creates a Dataset by transforming the data in the given location using a {@link DataFormat} either previously
    * supplied via {@link #dataFormat} or by default Json.
    *
    * @param location the location of the data file
    * @return the dataset
    * @throws IOException Something went wrong reading the data
    */
   public Dataset source(Resource location) throws IOException {
      if (dataFormat == null) {
         return type.read(location);
      }
      return type.read(location, dataFormat);
   }


}//END OF DatasetBuilder
