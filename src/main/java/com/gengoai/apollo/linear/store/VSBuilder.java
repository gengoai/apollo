package com.gengoai.apollo.linear.store;

import com.gengoai.apollo.linear.NDArray;

import java.util.stream.Stream;

/**
 * Abstract base builder for {@link VectorStore}s.
 *
 * @author David B. Bracewell
 */
public interface VSBuilder {

   /**
    * Add a vector to the vector store. The key is expected to be the label of the vector
    *
    * @param vector the vector to add
    * @return this vector store builder
    * @throws NullPointerException if the vector or its key is null
    */
   default VSBuilder add(NDArray vector) {
      return add(vector.getLabel(), vector);
   }

   /**
    * Add a vector to the vector store using the given key.
    *
    * @param key    the key to use for the vector
    * @param vector the vector to add
    * @return the vector store builder
    * @throws NullPointerException if the vector or its key is null
    */
   VSBuilder add(String key, NDArray vector);

   /**
    * Adds all vectors in the given iterable to the vector store
    *
    * @param vectors the vectors to add
    * @return this vector store builder
    * @throws NullPointerException if the vectors or their keys are null
    */
   default VSBuilder addAll(Iterable<NDArray> vectors) {
      vectors.forEach(this::add);
      return this;
   }

   /**
    * Adds all vectors in the given stream to the vector store
    *
    * @param vectors the vectors to add
    * @return this vector store builder
    * @throws NullPointerException if the vectors or their keys are null
    */
   default VSBuilder addAll(Stream<NDArray> vectors) {
      vectors.forEach(this::add);
      return this;
   }

   /**
    * Finalizes the building of the {@link VectorStore}.
    *
    * @return the built vector store
    */
   VectorStore build();

}// END OF VectorStoreBuilder
