package com.davidbracewell.apollo.ml.embedding;

import com.davidbracewell.apollo.linalg.store.CosineSignature;
import com.davidbracewell.apollo.linalg.store.InMemoryLSH;
import com.davidbracewell.apollo.linalg.store.VectorStore;
import com.davidbracewell.apollo.ml.EncoderPair;
import lombok.NonNull;

import java.util.function.IntFunction;

/**
 * The type Faruqui retrofitting.
 *
 * @author David B. Bracewell
 */
public class FaruquiRetrofitting implements Retrofitting {

   private final IntFunction<VectorStore<String>> vectorStoreCreator;

   /**
    * Instantiates a new Faruqui retrofitting.
    */
   public FaruquiRetrofitting() {
      this((dimension) -> InMemoryLSH.builder()
                                     .dimension(dimension)
                                     .signatureSupplier(CosineSignature::new)
                                     .createVectorStore());
   }

   /**
    * Instantiates a new Faruqui retrofitting.
    *
    * @param vectorStoreCreator the vector store creator
    */
   public FaruquiRetrofitting(@NonNull IntFunction<VectorStore<String>> vectorStoreCreator) {
      this.vectorStoreCreator = vectorStoreCreator;
   }

   @Override
   public Embedding process(@NonNull Embedding embedding) {
      EncoderPair encoderPair = embedding.getEncoderPair();
      VectorStore<String> origVectors = embedding.getVectorStore();

      //Do cool stuff

      VectorStore<String> newVectors = vectorStoreCreator.apply(origVectors.dimension());
      return new Embedding(encoderPair, newVectors);
   }

}//END OF FaruquiRetrofitting
