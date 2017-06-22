package com.davidbracewell.apollo.ml;

import com.davidbracewell.apollo.linalg.Vector;
import com.davidbracewell.apollo.linalg.VectorComposition;
import com.davidbracewell.apollo.ml.embedding.Embedding;
import com.davidbracewell.conversion.Cast;
import com.davidbracewell.function.SerializableSupplier;

import java.io.Serializable;
import java.util.stream.Collectors;

/**
 * @author David B. Bracewell
 */
public class EmbeddingVectorizer implements Vectorizer, Serializable {
   private static final long serialVersionUID = 1L;
   private final SerializableSupplier<Embedding> embeddingSupplier;
   private final VectorComposition composition;
   private volatile transient Embedding embedding;
   private EncoderPair encoderPair;

   public EmbeddingVectorizer(SerializableSupplier<Embedding> embeddingSupplier, VectorComposition composition) {
      this.embeddingSupplier = embeddingSupplier;
      this.composition = composition;
   }

   @Override
   public Vector apply(Example example) {
      Embedding embedding = getEmbedding();
      Vector v = composition.compose(embedding.getDimension(), example.getFeatureSpace()
                                                                      .filter(embedding::contains)
                                                                      .map(embedding::getVector)
                                                                      .collect(Collectors.toList()));
      if (example instanceof Instance) {
         Instance ii = Cast.as(example);
         v.setLabel(encoderPair.encodeLabel(ii.getLabel()));
         v.setWeight(ii.getWeight());
      }
      return v;
   }

   Embedding getEmbedding() {
      if (embedding == null) {
         synchronized (this) {
            if (embedding == null) {
               embedding = embeddingSupplier.get();
            }
         }
      }
      return embedding;
   }

   @Override
   public int getOutputDimension() {
      return getEmbedding().getDimension();
   }

   @Override
   public void setEncoderPair(EncoderPair encoderPair) {
      this.encoderPair = encoderPair;
   }

}// END OF EmbeddingVectorizer
