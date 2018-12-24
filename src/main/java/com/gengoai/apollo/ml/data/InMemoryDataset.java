package com.gengoai.apollo.ml.data;

import com.gengoai.apollo.ml.Example;
import com.gengoai.stream.MStream;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Iterator;
import java.util.List;

/**
 * The type In memory dataset.
 *
 * @author David B. Bracewell
 */
public class InMemoryDataset extends Dataset {
   private static final long serialVersionUID = 1L;
   private final List<Example> examples;

   /**
    * Instantiates a new In memory dataset.
    *
    * @param examples the examples
    */
   public InMemoryDataset(Collection<Example> examples) {
      super(DatasetType.InMemory);
      this.examples = new ArrayList<>(examples);
   }

   @Override
   protected void addAll(MStream<Example> stream) {
      stream.forEachLocal(examples::add);
   }

   @Override
   public Dataset cache() {
      return this;
   }

   @Override
   public void close() throws Exception {

   }

   @Override
   public Iterator<Example> iterator() {
      return examples.iterator();
   }

   @Override
   protected Dataset newSimilarDataset(MStream<Example> instances) {
      return new InMemoryDataset(instances.map(e -> e.copy()).collect());
   }

   @Override
   public int size() {
      return examples.size();
   }


   @Override
   public MStream<Example> stream() {
      return getStreamingContext().stream(examples);
   }

}//END OF InMemoryDataset
