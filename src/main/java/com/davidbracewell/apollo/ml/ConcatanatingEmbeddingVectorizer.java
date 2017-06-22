package com.davidbracewell.apollo.ml;

import com.davidbracewell.apollo.linalg.Vector;
import com.davidbracewell.apollo.ml.embedding.Embedding;
import com.davidbracewell.apollo.ml.sequence.Sequence;
import com.davidbracewell.collection.Collect;
import com.davidbracewell.conversion.Cast;
import com.davidbracewell.function.SerializableSupplier;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

/**
 * @author David B. Bracewell
 */
public class ConcatanatingEmbeddingVectorizer implements Vectorizer, Serializable {
   private static final long serialVersionUID = 1L;
   private final SerializableSupplier<Embedding> embeddingSupplier;
   private final int maxWidth;
   private volatile transient Embedding embedding;
   private int outputDimension;
   private EncoderPair encoderPair;

   public ConcatanatingEmbeddingVectorizer(SerializableSupplier<Embedding> embeddingSupplier, int maxWidth) {
      this.embeddingSupplier = embeddingSupplier;
      this.maxWidth = maxWidth;
   }


   @Override
   public Vector apply(Example example) {
      Embedding embedding = getEmbedding();
      Vector v = Vector.dZeros(getOutputDimension());

      final List<Feature> features;
      final Object label;
      final double weight;
      if (example instanceof Instance) {
         Instance instance = Cast.as(example);
         features = instance.getFeatures();
         label = instance.getLabel();
         weight = instance.getWeight();
      } else {
         features = new ArrayList<>();
         Sequence sequence = Cast.as(example);
         sequence.asInstances().forEach(i -> features.add(i.getFeatures().get(0)));
         label = sequence.get(sequence.size() - 1).getLabel();
         weight = 1.0;
      }

      for (int i = 0; i < Math.min(features.size(), maxWidth); i++) {
         Vector e = embedding.getVector(features.get(i).getName());
         for (Vector.Entry entry : Collect.asIterable(e.nonZeroIterator())) {
            v.set(i * embedding.getDimension() + entry.index, entry.value);
         }
      }

      v.setLabel(label);
      v.setWeight(weight);
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
      return outputDimension;
   }

   @Override
   public void setEncoderPair(EncoderPair encoderPair) {
      this.encoderPair = encoderPair;
      this.outputDimension = maxWidth * getEmbedding().getDimension() + getEmbedding().getDimension();
   }

}// END OF EmbeddingVectorizer
