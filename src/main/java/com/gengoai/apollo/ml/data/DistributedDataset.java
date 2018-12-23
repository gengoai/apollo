package com.gengoai.apollo.ml.data;

import com.gengoai.apollo.ml.Example;
import com.gengoai.function.SerializableFunction;
import com.gengoai.stream.MStream;
import com.gengoai.stream.StreamingContext;

import java.util.Iterator;
import java.util.Random;

/**
 * @author David B. Bracewell
 */
public class DistributedDataset extends Dataset {
   private MStream<Example> stream = StreamingContext.distributed().empty();

   public DistributedDataset() {

   }

   protected DistributedDataset(MStream<Example> stream) {
      this.stream = stream;
   }

   @Override
   public Dataset cache() {
      this.stream = stream.cache();
      return this;
   }

   @Override
   protected void addAll(MStream<Example> stream) {
      this.stream = this.stream.union(stream);
   }

   @Override
   public void close() throws Exception {
      stream.close();
   }

   @Override
   public DatasetType getType() {
      return DatasetType.Distributed;
   }

   @Override
   public Iterator<Example> iterator() {
      return stream.iterator();
   }

   @Override
   public Dataset mapSelf(SerializableFunction<? super Example, ? extends Example> function) {
      this.stream = stream.map(function);
      return this;
   }

   @Override
   protected Dataset newSimilarDataset(MStream<Example> instances) {
      return new DistributedDataset(instances);
   }

   @Override
   public Dataset shuffle(Random random) {
      return new DistributedDataset(stream.shuffle(random));
   }
}//END OF DistributedDataset
