package com.gengoai.apollo.ml.data;

import com.gengoai.apollo.ml.Example;
import com.gengoai.stream.MStream;

/**
 * <p>A dataset wrapping a Mango Stream.</p>
 *
 * @author David B. Bracewell
 */
public class MStreamDataset extends BaseStreamDataset {
   private static final long serialVersionUID = 1L;

   public MStreamDataset(MStream<Example> stream) {
      super(DatasetType.Stream, stream);
   }

   @Override
   protected Dataset newSimilarDataset(MStream<Example> instances) {
      return new MStreamDataset(instances);
   }

}//END OF MStreamDataset
