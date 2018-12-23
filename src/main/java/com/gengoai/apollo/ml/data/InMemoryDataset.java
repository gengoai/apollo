package com.gengoai.apollo.ml.data;

import com.gengoai.apollo.ml.Example;
import com.gengoai.function.SerializableFunction;
import com.gengoai.stream.MStream;

import java.util.*;

/**
 * @author David B. Bracewell
 */
public class InMemoryDataset extends Dataset {
   private static final long serialVersionUID = 1L;
   private final List<Example> examples;

   public InMemoryDataset() {
      this.examples = new ArrayList<>();
   }

   public InMemoryDataset(Collection<Example> examples) {
      this.examples = new ArrayList<>(examples);
   }

   @Override
   protected void addAll(MStream<Example> stream) {
      stream.forEachLocal(examples::add);
   }

   @Override
   public void close() throws Exception {

   }

   @Override
   public Dataset cache() {
      return this;
   }

   @Override
   public DatasetType getType() {
      return DatasetType.InMemory;
   }

   @Override
   public Dataset mapSelf(SerializableFunction<? super Example, ? extends Example> function) {
      for (int i = 0; i < examples.size(); i++) {
         examples.set(i, function.apply(examples.get(i)));
      }
      return this;
   }

   @Override
   public Iterator<Example> iterator() {
      return examples.iterator();
   }

   @Override
   protected Dataset newSimilarDataset(MStream<Example> instances) {
      Dataset ds = new InMemoryDataset();
      ds.addAll(instances.map(Example::copy));
      return ds;
   }

   @Override
   public Dataset shuffle(Random random) {
      Collections.shuffle(examples);
      return this;
   }

}//END OF InMemoryDataset
