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
import com.gengoai.apollo.ml.Split;
import com.gengoai.collection.counter.Counter;
import com.gengoai.conversion.Cast;
import com.gengoai.function.SerializableFunction;
import com.gengoai.stream.MStream;

import java.util.Iterator;
import java.util.Random;
import java.util.stream.Stream;

import static com.gengoai.Validation.checkArgument;

/**
 * <p>
 * Abstract base {@link ExampleDataset} backed by an <code>MStream</code>.
 * </p>
 *
 * @author David B. Bracewell
 */
class StreamBasedExampleDataset extends ExampleDataset {
   private static final long serialVersionUID = 1L;
   private final DatasetType datasetType;
   private MStream<Example> stream;

   /**
    * Instantiates a new Base stream dataset.
    *
    * @param datasetType the dataset type
    * @param stream      the stream
    */
   public StreamBasedExampleDataset(DatasetType datasetType,
                                    MStream<Example> stream) {
      this.datasetType = datasetType;
      this.stream = stream == null
                    ? datasetType.getStreamingContext().empty()
                    : stream;
   }

   @Override
   public Iterator<ExampleDataset> batchIterator(int batchSize) {
      return stream.partition(batchSize)
                   .map(batch -> (ExampleDataset) datasetOf(toMStream(batch).map(Example::copy)))
                   .iterator();
   }

   @Override
   public ExampleDataset cache() {
      stream.cache();
      return this;
   }

   @Override
   public void close() throws Exception {
      stream.close();
   }

   protected StreamBasedExampleDataset datasetOf(MStream<Example> examples) {
      return new StreamBasedExampleDataset(datasetType, examples);
   }

   @Override
   public Split[] fold(int numberOfFolds) {
      checkArgument(numberOfFolds > 0, "Number of folds must be >= 0");
      checkArgument(size() >= numberOfFolds, "Number of folds must be <= number of examples");
      Split[] folds = new Split[numberOfFolds];
      long foldSize = size() / numberOfFolds;
      for(int i = 0; i < numberOfFolds; i++) {
         long testStart = i * foldSize;
         long testEnd = testStart + foldSize;
         MStream<Example> testStream = slice(testStart, testEnd).stream();
         MStream<Example> trainStream = getStreamingContext().empty();
         if(testStart > 0) {
            trainStream = trainStream.union(slice(0, testStart).stream());
         }
         if(testEnd < size()) {
            trainStream = trainStream.union(slice(testEnd, size()).stream());
         }
         folds[i] = new Split(datasetOf(trainStream), datasetOf(testStream));
      }
      return folds;
   }

   @Override
   public DatasetType getType() {
      return datasetType;
   }

   @Override
   public Iterator<Example> iterator() {
      return stream.iterator();
   }

   @Override
   public ExampleDataset map(SerializableFunction<? super Example, ? extends Example> function) {
      return datasetOf(stream.map(function));
   }

   @Override
   public ExampleDataset sample(boolean withReplacement, int sampleSize) {
      checkArgument(sampleSize > 0, "Sample size must be > 0");
      return datasetOf(stream().sample(withReplacement, sampleSize).map(e -> Cast.as(e.copy())));
   }

   @Override
   public ExampleDataset shuffle(Random random) {
      return datasetOf(stream.shuffle(random));
   }

   @Override
   public ExampleDataset slice(long start, long end) {
      return datasetOf(stream().skip(start).limit(end - start));
   }

   @Override
   public Split split(double pctTrain) {
      checkArgument(pctTrain > 0 && pctTrain < 1, "Percentage should be between 0 and 1");
      int split = (int) Math.floor(pctTrain * size());
      return new Split(slice(0, split), slice(split, size()));
   }

   @Override
   public MStream<Example> stream() {
      return stream;
   }

   protected MStream<Example> toMStream(Stream<Example> stream) {
      return datasetType.getStreamingContext().stream(stream);
   }

   @Override
   public MStream<Example> parallelStream() {
      return stream.parallel();
   }


   @Override
   public ExampleDataset oversample() {
      Counter<String> fCount = calculateClassDistribution();
      int targetCount = (int) fCount.maximumCount();

      StreamBasedExampleDataset dataset = datasetOf(getStreamingContext().empty());

      for(Object label : fCount.items()) {
         MStream<Example> fStream = stream()
               .filter(e -> e.getLabelSpace().anyMatch(label::equals))
               .cache();
         int count = (int) fStream.count();
         int curCount = 0;
         while(curCount + count < targetCount) {
            dataset.stream.union(fStream);
            curCount += count;
         }
         if(curCount < targetCount) {
            dataset.stream.union(fStream.sample(false, targetCount - curCount));
         }
         else if(count == targetCount) {
            dataset.stream.union(fStream);
         }
      }
      return dataset;
   }

   @Override
   public ExampleDataset undersample() {
      Counter<String> fCount = calculateClassDistribution();
      int targetCount = (int) fCount.minimumCount();
      StreamBasedExampleDataset dataset = datasetOf(getStreamingContext().empty());
      for(Object label : fCount.items()) {
         dataset.stream.union(stream().filter(e -> e.getLabelSpace().anyMatch(label::equals))
                                      .sample(false, targetCount));
      }
      return dataset;
   }

}//END OF BaseStreamDataset
