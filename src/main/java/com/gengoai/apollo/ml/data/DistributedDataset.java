package com.gengoai.apollo.ml.data;

import com.gengoai.apollo.ml.Example;
import com.gengoai.stream.MStream;

/**
 * The type Distributed dataset.
 *
 * @author David B. Bracewell
 */
public class DistributedDataset extends BaseStreamDataset {
   private static final long serialVersionUID = 1L;

   /**
    * Instantiates a new Distributed dataset.
    *
    * @param stream the stream
    */
   public DistributedDataset(MStream<Example> stream) {
      super(DatasetType.Distributed, stream);
   }

   @Override
   protected Dataset newSimilarDataset(MStream<Example> instances) {
      return new DistributedDataset(instances);
   }

}//END OF DistributedDataset
