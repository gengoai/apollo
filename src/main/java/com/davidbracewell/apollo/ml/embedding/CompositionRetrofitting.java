package com.davidbracewell.apollo.ml.embedding;

import com.davidbracewell.apollo.linalg.Vector;
import com.davidbracewell.apollo.linalg.store.VectorStore;
import com.davidbracewell.io.resource.Resource;
import com.google.common.base.Throwables;
import lombok.Getter;
import lombok.NonNull;
import lombok.Setter;

/**
 * The type Composition retrofitting.
 *
 * @author David B. Bracewell
 */
public class CompositionRetrofitting implements Retrofitting {
   private final Embedding background;
   @Getter
   @Setter
   private int neighborSize = 0;
   @Getter
   @Setter
   private double neighborWeight = 0;
   @Getter
   @Setter
   private double neighborThreshold = 0;

   /**
    * Instantiates a new Composition retrofitting.
    *
    * @param background the background
    */
   public CompositionRetrofitting(@NonNull Resource background) {
      try {
         this.background = Embedding.read(background);
      } catch (Exception e) {
         throw Throwables.propagate(e);
      }
   }

   @Override
   public Embedding process(@NonNull Embedding embedding) {
      VectorStore<String> newEmbedding = embedding
                                            .getVectorStore()
                                            .createNew();
      embedding
         .getVocab()
         .forEach(term -> {
            Vector tv = embedding.getVector(term).copy();
            if (background.contains(term)) {
               tv.addSelf(background.getVector(term));
               if (neighborSize > 0 && neighborWeight > 0) {
                  background
                     .nearest(term, neighborSize, neighborThreshold)
                     .forEach(n -> tv.addSelf(n.mapMultiply(neighborWeight)));
               }
            }
            newEmbedding.add(term, tv);
         });

      return new Embedding(embedding.getEncoderPair(), newEmbedding);
   }
}// END OF CompositionRetrofitting
