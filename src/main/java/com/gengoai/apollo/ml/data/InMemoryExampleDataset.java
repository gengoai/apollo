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

package com.gengoai.apollo.ml.data;

import com.gengoai.Validation;
import com.gengoai.apollo.ml.Example;
import com.gengoai.apollo.ml.Split;
import com.gengoai.function.SerializableFunction;
import com.gengoai.stream.MStream;
import com.gengoai.stream.StreamingContext;
import lombok.NonNull;

import java.util.*;
import java.util.stream.Collectors;

public class InMemoryExampleDataset extends ExampleDataset {
   private final List<Example> examples = new ArrayList<>();

   public InMemoryExampleDataset(final Collection<Example> examples) {
      this.examples.addAll(examples);
   }

   public InMemoryExampleDataset(final MStream<Example> examples) {
      examples.forEach(this.examples::add);
   }


   @Override
   public Split[] fold(int numberOfFolds) {
      return new Split[0];
   }

   @Override
   public ExampleDataset map(@NonNull SerializableFunction<? super Example, ? extends Example> function) {
      return new InMemoryExampleDataset(examples.parallelStream()
                                                .map(function)
                                                .collect(Collectors.toList()));
   }

   @Override
   public ExampleDataset oversample() {
      return null;
   }

   @Override
   public ExampleDataset sample(boolean withReplacement, int sampleSize) {
      return null;
   }

   @Override
   public ExampleDataset shuffle(@NonNull Random random) {
      Collections.shuffle(examples, random);
      return this;
   }

   @Override
   public DatasetType getType() {
      return DatasetType.InMemory;
   }

   @Override
   public Iterator<ExampleDataset> batchIterator(int batchSize) {
      Validation.checkArgument(batchSize > 0, "Batch size must be > 0");
      return new Iterator<>() {
         int index = 0;

         @Override
         public boolean hasNext() {
            return index < examples.size();
         }

         @Override
         public ExampleDataset next() {
            ExampleDataset next = slice(index, Math.min(index + batchSize, examples.size()));
            index = index + batchSize;
            return next;
         }
      };
   }

   @Override
   public ExampleDataset cache() {
      return this;
   }

   @Override
   public MStream<Example> stream() {
      return StreamingContext.local().stream(examples);
   }

   @Override
   public MStream<Example> parallelStream() {
      return stream().parallel();
   }

   @Override
   public ExampleDataset slice(long start, long end) {
      return new InMemoryExampleDataset(examples.subList((int) start, (int) end));
   }

   @Override
   public Split split(double pctTrain) {
      int splitIndex = (int) (pctTrain * examples.size());
      return new Split(slice(0, splitIndex), slice(splitIndex, examples.size()));
   }

   @Override
   public ExampleDataset undersample() {
      return null;
   }

   @Override
   public void close() throws Exception {

   }

   @Override
   public Iterator<Example> iterator() {
      return examples.iterator();
   }

}//END OF InMemoryDataset
