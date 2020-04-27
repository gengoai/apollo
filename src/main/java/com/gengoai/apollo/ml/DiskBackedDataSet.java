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

import com.gengoai.collection.disk.DiskMap;
import com.gengoai.function.SerializableFunction;
import com.gengoai.io.resource.Resource;
import com.gengoai.stream.MStream;
import com.gengoai.stream.StreamingContext;
import com.gengoai.stream.Streams;
import lombok.NonNull;

import java.util.*;
import java.util.concurrent.atomic.AtomicLong;

import static com.gengoai.Validation.checkArgument;

/**
 * @author David B. Bracewell
 */
public class DiskBackedDataSet extends DataSet {
   private static final long serialVersionUID = 1L;
   private DiskMap<Long, Datum> map;
   private List<Long> indexes = new ArrayList<>();

   public DiskBackedDataSet(@NonNull Resource location, @NonNull MStream<Datum> stream) {
      this.map = DiskMap.<Long, Datum>builder()
            .namespace("data")
            .compressed(true)
            .file(location)
            .build();
      AtomicLong id = new AtomicLong();
      stream.forEach(d -> {
         long did = id.getAndIncrement();
         indexes.add(did);
         map.put(did, d);
      });
   }

   private DiskBackedDataSet(DiskMap<Long, Datum> m,
                             List<Long> i) {
      this.map = m;
      this.indexes = i;
   }

   @Override
   public Iterator<DataSet> batchIterator(int batchSize) {
      return new Iterator<DataSet>() {
         long position = 0;

         @Override
         public boolean hasNext() {
            return position < indexes.size();
         }

         @Override
         public DataSet next() {
            long start = position;
            long end = Math.min(indexes.size(), position + batchSize);
            position += batchSize;
            return slice(start, end);
         }
      };
   }

   @Override
   public DataSet cache() {
      return this;
   }

   private DataSet create(List<Long> indexes) {
      DataSet ds = new DiskBackedDataSet(map, indexes);
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
         folds[i] = new Split(new StreamingDataSet(trainStream), new StreamingDataSet(testStream));
         folds[i].train.putAllMetadata(getMetadata());
         folds[i].test.putAllMetadata(getMetadata());
      }
      return folds;
   }

   @Override
   public DataSetType getType() {
      return DataSetType.OnDisk;
   }

   @Override
   public Iterator<Datum> iterator() {
      return new Iterator<Datum>() {
         int index = 0;

         @Override
         public boolean hasNext() {
            return index < indexes.size();
         }

         @Override
         public Datum next() {
            long l = indexes.get(index);
            index++;
            return map.get(l);
         }
      };
   }

   @Override
   public DataSet map(@NonNull SerializableFunction<? super Datum, ? extends Datum> function) {
      indexes.parallelStream()
             .forEach(i -> {
                Datum d = function.apply(map.get(i));
                map.put(i, d);
             });
      return this;
   }

   @Override
   public DataSet oversample(@NonNull String observationName) {
      DataSet ds = new StreamingDataSet(parallelStream()).oversample(observationName);
      ds.putAllMetadata(getMetadata());
      return ds;
   }

   @Override
   public MStream<Datum> parallelStream() {
      return stream().parallel();
   }

   @Override
   public DataSet sample(boolean withReplacement, int sampleSize) {
      DataSet ds = new StreamingDataSet(parallelStream()).sample(withReplacement, sampleSize);
      ds.putAllMetadata(getMetadata());
      return ds;
   }

   @Override
   public DataSet shuffle(Random random) {
      Collections.shuffle(indexes);
      return this;
   }

   @Override
   public long size() {
      return indexes.size();
   }

   @Override
   public DataSet slice(long start, long end) {
      return create(indexes.subList((int) start, (int) end));
   }

   @Override
   public Split split(double pctTrain) {
      checkArgument(pctTrain > 0 && pctTrain < 1, "Percentage should be between 0 and 1");
      int split = (int) Math.floor(pctTrain * indexes.size());
      List<Long> i1 = indexes.subList(0, split);
      List<Long> i2 = indexes.subList(split, indexes.size());
      return new Split(create(i1), create(i2));
   }

   @Override
   public MStream<Datum> stream() {
      return StreamingContext.local().stream(Streams.reusableStream(indexes).map(map::get));
   }

   @Override
   public DataSet undersample(@NonNull String observationName) {
      DataSet ds = new StreamingDataSet(parallelStream()).undersample(observationName);
      ds.putAllMetadata(getMetadata());
      return ds;
   }
}//END OF DiskBackedDataSet
