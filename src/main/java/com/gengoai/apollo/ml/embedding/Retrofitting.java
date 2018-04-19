package com.gengoai.apollo.ml.embedding;


import com.gengoai.apollo.linear.store.VectorStore;

/**
 * The interface Retrofitting.
 *
 * @author David B. Bracewell
 */
public interface Retrofitting {

   /**
    * Process embedding.
    *
    * @param embedding the embedding
    * @return the embedding
    */
   Embedding process(VectorStore<String> embedding);

}//END OF Retrofitting
