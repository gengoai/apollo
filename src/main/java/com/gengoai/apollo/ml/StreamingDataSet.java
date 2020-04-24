/*
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

package com.gengoai.apollo.ml;

import com.gengoai.Copyable;
import com.gengoai.apollo.ml.observation.Variable;
import com.gengoai.collection.counter.Counter;
import com.gengoai.conversion.Cast;
import com.gengoai.function.SerializableFunction;
import com.gengoai.stream.MStream;
import lombok.NonNull;

import java.util.Iterator;
import java.util.Random;

import static com.gengoai.Validation.checkArgument;

/**
 * <p>A {@link DataSet} backed by an MStream. </p>
 *
 * @author David B. Bracewell
 */
public class StreamingDataSet extends DataSet {
   private MStream<Datum> stream;

   /**
    * Instantiates a new Streaming data set.
    *
    * @param stream the stream
    */
   public StreamingDataSet(@NonNull MStream<Datum> stream) {
      this.stream = stream;
   }

   @Override
   public Iterator<DataSet> batchIterator(int batchSize) {
      return stream.partition(batchSize)
                   .map(batch -> (DataSet) datasetOf(getType().getStreamingContext().stream(batch).map(Datum::copy)))
                   .iterator();
   }

   @Override
   public DataSet cache() {
      InMemoryDataSet out = new InMemoryDataSet(stream.collect());
      out.metadata.putAll(Copyable.deepCopy(metadata));
      return out;
   }

   protected StreamingDataSet datasetOf(MStream<Datum> examples) {
      StreamingDataSet ds = new StreamingDataSet(examples.map(Datum::copy));
      ds.putAllMetadata(getMetadata());
      return ds;
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
         MStream<Datum> testStream = slice(testStart, testEnd).stream();
         MStream<Datum> trainStream = getStreamingContext().empty();
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
   public DataSetType getType() {
      if(stream.getContext().isDistributed()) {
         return DataSetType.Distributed;
      }
      return DataSetType.LocalStreaming;
   }

   @Override
   public Iterator<Datum> iterator() {
      return stream.iterator();
   }

   @Override
   public DataSet map(@NonNull SerializableFunction<? super Datum, ? extends Datum> function) {
      StreamingDataSet out = new StreamingDataSet(stream.map(function));
      out.metadata.putAll(Copyable.deepCopy(metadata));
      return out;
   }

   @Override
   public DataSet oversample(@NonNull String observationName) {
      Counter<String> fCount = calculateClassDistribution(observationName);
      int targetCount = (int) fCount.maximumCount();

      StreamingDataSet dataset = datasetOf(getStreamingContext().empty());

      for(Object label : fCount.items()) {
         MStream<Datum> fStream = stream()
               .filter(e -> e.get(observationName).getVariableSpace().map(Variable::getName).anyMatch(label::equals))
               .cache();
         int count = (int) fStream.count();
         int curCount = 0;
         while(curCount + count < targetCount) {
            dataset.stream.union(fStream);
            curCount += count;
         }
         if(curCount < targetCount) {
            dataset.stream.union(fStream.sample(false, targetCount - curCount));
         } else if(count == targetCount) {
            dataset.stream.union(fStream);
         }
      }
      return dataset;
   }

   @Override
   public MStream<Datum> parallelStream() {
      return stream.parallel();
   }

   @Override
   public DataSet sample(boolean withReplacement, int sampleSize) {
      checkArgument(sampleSize > 0, "Sample size must be > 0");
      return datasetOf(stream().sample(withReplacement, sampleSize).map(e -> Cast.as(e.copy())));
   }

   @Override
   public DataSet shuffle(Random random) {
      stream = stream.shuffle(random);
      return this;
   }

   @Override
   public long size() {
      return stream.count();
   }

   @Override
   public DataSet slice(long start, long end) {
      return datasetOf(stream().skip(start).limit(end - start));
   }

   @Override
   public Split split(double pctTrain) {
      checkArgument(pctTrain > 0 && pctTrain < 1, "Percentage should be between 0 and 1");
      int split = (int) Math.floor(pctTrain * size());
      return new Split(slice(0, split), slice(split, size()));
   }

   @Override
   public MStream<Datum> stream() {
      return stream;
   }

   @Override
   public DataSet undersample(@NonNull String observationName) {
      Counter<String> fCount = calculateClassDistribution(observationName);
      int targetCount = (int) fCount.minimumCount();
      StreamingDataSet dataset = datasetOf(getStreamingContext().empty());
      for(Object label : fCount.items()) {
         dataset.stream.union(stream().filter(e -> e.get(observationName)
                                                    .getVariableSpace()
                                                    .map(Variable::getName)
                                                    .anyMatch(label::equals))
                                      .sample(false, targetCount));
      }
      return dataset;
   }

}//END OF StreamingDataSet
