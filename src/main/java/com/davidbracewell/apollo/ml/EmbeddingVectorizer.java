package com.davidbracewell.apollo.ml;

import com.davidbracewell.apollo.linalg.DenseVector;
import com.davidbracewell.apollo.linalg.Vector;
import com.davidbracewell.apollo.linalg.VectorComposition;
import com.davidbracewell.apollo.ml.embedding.Embedding;

import java.io.Serializable;
import java.util.Iterator;

/**
 * @author David B. Bracewell
 */
public class EmbeddingVectorizer implements Vectorizer, Serializable {
   private static final long serialVersionUID = 1L;
   private final Embedding embedding;
   private final VectorComposition composition;

   public EmbeddingVectorizer(Embedding embedding, VectorComposition composition) {
      this.embedding = embedding;
      this.composition = composition;
   }

   @Override
   public Vector apply(Example example) {
      Vector vPrime = DenseVector.zeros(embedding.getDimension());
      for (Iterator<Vector> itr = example.getFeatureSpace()
                                         .filter(embedding::contains)
                                         .map(embedding::getVector).iterator(); itr.hasNext(); ) {
         vPrime = composition.compose(embedding.getDimension(), vPrime, itr.next());
      }
      return vPrime;
   }

   @Override
   public int getOutputDimension() {
      return embedding.getDimension();
   }

   @Override
   public void setEncoderPair(EncoderPair encoderPair) {
   }

}// END OF EmbeddingVectorizer
