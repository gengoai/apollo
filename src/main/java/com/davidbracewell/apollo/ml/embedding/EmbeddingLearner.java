package com.davidbracewell.apollo.ml.embedding;

import com.davidbracewell.apollo.ml.Learner;
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



}// END OF EmbeddingLearner
