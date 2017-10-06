package com.davidbracewell.apollo.ml.embedding;

import com.davidbracewell.apollo.ml.Learner;
import com.davidbracewell.apollo.ml.data.Dataset;
import com.davidbracewell.apollo.ml.sequence.Sequence;
import lombok.Getter;
import lombok.Setter;

/**
 * <p>Base class for learners that build embedding models.</p>
 *
 * @author David B. Bracewell
 */
public abstract class EmbeddingLearner extends Learner<Sequence, Embedding> {
   private static final long serialVersionUID = 1L;

   @Getter
   @Setter
   private int dimension = 300;


   @Override
   public Embedding train(Dataset<Sequence> dataset) {
      dataset.encode();
      Embedding model = trainImpl(dataset);
      model.finishTraining();
      return model;
   }

}// END OF EmbeddingLearner
