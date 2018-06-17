package com.gengoai.apollo.ml.embedding;

import com.gengoai.apollo.linear.NDArray;
import com.gengoai.apollo.linear.store.VectorStore;
import com.gengoai.apollo.linear.store.VectorStoreBuilder;
import com.gengoai.io.resource.Resource;
import lombok.Getter;
import lombok.NonNull;
import lombok.Setter;

import java.io.IOException;

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
         throw new RuntimeException(e);
      }
   }

   @Override
   public Embedding process(@NonNull VectorStore<String> embedding) {
      VectorStoreBuilder<String> newEmbedding = embedding.toBuilder();
      embedding
         .keys()
         .forEach(term -> {
            NDArray tv = embedding.get(term).copy();
            if (background.contains(term)) {
               tv.addi(background.get(term));
               if (neighborSize > 0 && neighborWeight > 0) {
                  background
                     .nearest(term, neighborSize, neighborThreshold)
                     .forEach(n -> tv.addi(n.mul(neighborWeight)));
               }
            }
            newEmbedding.add(term, tv);
         });

      try {
         return new Embedding(newEmbedding.build());
      } catch (IOException e) {
         throw new RuntimeException(e);
      }
   }
}// END OF CompositionRetrofitting
