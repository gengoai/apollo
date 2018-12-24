package com.gengoai.apollo.ml.data;

import com.gengoai.apollo.ml.Example;
import com.gengoai.stream.MStream;

import java.util.Iterator;

/**
 * @author David B. Bracewell
 */
public abstract class BaseStreamDataset extends Dataset {
   private static final long serialVersionUID = 1L;
   protected MStream<Example> stream;

   public BaseStreamDataset(DatasetType datasetType, MStream<Example> stream) {
      super(datasetType);
      this.stream = stream == null
                    ? datasetType.getStreamingContext().empty()
                    : stream;
   }

   @Override
   protected void addAll(MStream<Example> stream) {
      this.stream = this.stream.union(stream);
   }

   @Override
   public Dataset cache() {
      if (getStreamingContext().isDistributed()) {
         return newSimilarDataset(stream.cache());
      }
      return new InMemoryDataset(stream.collect());
   }

   @Override
   public void close() throws Exception {
      stream.close();
   }

   @Override
   public Iterator<Example> iterator() {
      return stream.iterator();
   }


   @Override
   public MStream<Example> stream() {
      return stream;
   }
}//END OF BaseStreamDataset
