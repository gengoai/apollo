package com.davidbracewell.apollo.ml.embedding;

import com.davidbracewell.apollo.ml.Learner;
import com.davidbracewell.apollo.ml.sequence.Sequence;

/**
 * <p>Base class for learners that build embedding models.</p>
 *
 * @author David B. Bracewell
 */
public abstract class EmbeddingLearner extends Learner<Sequence, Embedding> {
   private static final long serialVersionUID = 1L;
}// END OF EmbeddingLearner
