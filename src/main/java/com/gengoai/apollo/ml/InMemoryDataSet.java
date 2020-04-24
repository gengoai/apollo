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

import com.gengoai.Validation;
import com.gengoai.apollo.ml.observation.Variable;
import com.gengoai.collection.counter.Counter;
import com.gengoai.function.SerializableFunction;
import com.gengoai.stream.MStream;
import com.gengoai.stream.StreamingContext;
import lombok.NonNull;

import java.util.*;
import java.util.stream.IntStream;

import static com.gengoai.Validation.checkArgument;

/**
 * <p>
 * A {@link DataSet} implementation where data is stored in memory. All operations modify the underlying collection of
 * datum.
 * </p>
 */
public class InMemoryDataSet extends DataSet {
   private static final long serialVersionUID = 1L;
   private final List<Datum> data = new ArrayList<>();

   /**
    * Instantiates a new In memory data set.
    *
    * @param data the data
    */
   public InMemoryDataSet(@NonNull Collection<? extends Datum> data) {
      this.data.addAll(data);
   }

   @Override
   public Iterator<DataSet> batchIterator(int batchSize) {
      Validation.checkArgument(batchSize > 0, "Batch size must be > 0");
      return new Iterator<>() {
         int index = 0;

         @Override
         public boolean hasNext() {
            return index < data.size();
         }

         @Override
         public DataSet next() {
            DataSet next = slice(index, Math.min(index + batchSize, data.size()));
            index = index + batchSize;
            return next;
         }
      };
   }

   @Override
   public DataSet cache() {
      return this;
   }

   protected InMemoryDataSet emptyWithMetadata() {
      InMemoryDataSet ds = new InMemoryDataSet(Collections.emptyList());
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
         InMemoryDataSet test = emptyWithMetadata();
         subListCopy(testStart, testEnd, test);
         InMemoryDataSet train = emptyWithMetadata();
         if(testStart > 0) {
            subListCopy(0, testStart, train);
         }
         if(testEnd < size()) {
            subListCopy(testEnd, data.size(), train);
         }
         folds[i] = new Split(train, test);
      }
      return folds;
   }

   @Override
   public DataSetType getType() {
      return DataSetType.InMemory;
   }

   @Override
   public Iterator<Datum> iterator() {
      return data.iterator();
   }

   @Override
   public DataSet map(SerializableFunction<? super Datum, ? extends Datum> function) {
      IntStream.range(0, data.size())
               .forEach(i -> data.set(i, function.apply(data.get(i))));
      return this;
   }

   @Override
   public DataSet oversample(@NonNull String observationName) {
      Counter<String> fCount = calculateClassDistribution(observationName);
      int targetCount = (int) fCount.maximumCount();

      InMemoryDataSet dataset = emptyWithMetadata();

      for(Object label : fCount.items()) {
         MStream<Datum> fStream = stream()
               .filter(e -> e.get(observationName).getVariableSpace().map(Variable::getName).anyMatch(label::equals))
               .map(Datum::copy)
               .cache();

         int count = (int) fStream.count();
         int curCount = 0;

         while(curCount + count < targetCount) {
            fStream.forEach(dataset.data::add);
            curCount += count;
         }

         if(curCount < targetCount) {
            fStream.sample(false, targetCount - curCount)
                   .forEach(dataset.data::add);
         } else if(count == targetCount) {
            fStream.forEach(dataset.data::add);
         }
      }
      return dataset;
   }

   @Override
   public MStream<Datum> parallelStream() {
      return StreamingContext.local().stream(this).parallel();
   }

   @Override
   public DataSet sample(boolean withReplacement, int sampleSize) {
      InMemoryDataSet ds = emptyWithMetadata();
      stream().sample(withReplacement, sampleSize)
              .map(Datum::copy)
              .forEach(ds.data::add);
      return ds;
   }

   @Override
   public DataSet shuffle() {
      Collections.shuffle(data);
      return this;
   }

   @Override
   public DataSet shuffle(Random random) {
      Collections.shuffle(data, random);
      return this;
   }

   @Override
   public long size() {
      return data.size();
   }

   @Override
   public DataSet slice(long start, long end) {
      InMemoryDataSet dataset = emptyWithMetadata();
      subListCopy(start, end, dataset);
      return dataset;
   }

   @Override
   public Split split(double pctTrain) {
      checkArgument(pctTrain > 0 && pctTrain < 1, "Percentage should be between 0 and 1");
      int split = (int) Math.floor(pctTrain * size());
      InMemoryDataSet train = emptyWithMetadata();
      subListCopy(0, split, train);
      InMemoryDataSet test = emptyWithMetadata();
      subListCopy(split, data.size(), test);
      return new Split(train, test);
   }

   @Override
   public MStream<Datum> stream() {
      return StreamingContext.local().stream(this);
   }

   private void subListCopy(long start, long end, InMemoryDataSet target) {
      for(int i = (int) start; i < Math.min(end, data.size()); i++) {
         target.data.add(data.get(i).copy());
      }
   }

   @Override
   public DataSet undersample(@NonNull String observationName) {
      Counter<String> fCount = calculateClassDistribution(observationName);
      int targetCount = (int) fCount.minimumCount();
      InMemoryDataSet dataset = emptyWithMetadata();
      for(Object label : fCount.items()) {
         stream().filter(e -> e.get(observationName)
                               .getVariableSpace()
                               .map(Variable::getName)
                               .anyMatch(label::equals))
                 .sample(false, targetCount)
                 .map(Datum::copy)
                 .forEach(dataset.data::add);
      }
      return dataset;
   }

}//END OF InMemoryDataset
