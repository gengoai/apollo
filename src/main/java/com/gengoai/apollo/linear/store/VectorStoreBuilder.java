package com.gengoai.apollo.linear.store;

import com.gengoai.apollo.linear.NDArray;
import com.gengoai.apollo.stat.measure.Measure;
import com.gengoai.apollo.stat.measure.Similarity;

import java.io.IOException;

/**
 * Abstract base builder for {@link VectorStore}s.
 *
 * @author David B. Bracewell
 */
public abstract class VectorStoreBuilder {
   protected int dimension = -1;
   private Measure measure = Similarity.Cosine;


   /**
    * Add a vector to the vector store. The key is expected to be the label of the vector
    *
    * @param vector the vector to add
    * @return this vector store builder
    * @throws NullPointerException if the vector or its key is null
    */
   public final VectorStoreBuilder add(NDArray vector) {
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
   public abstract VectorStoreBuilder add(String key, NDArray vector);

   /**
    * Adds all vectors in the given iterable to the vector store
    *
    * @param vectors the vectors to add
    * @return this vector store builder
    * @throws NullPointerException if the vectors or their keys are null
    */
   public final VectorStoreBuilder addAll(Iterable<NDArray> vectors) {
      vectors.forEach(this::add);
      return this;
   }

   /**
    * Finalizes the building of the {@link VectorStore}.
    *
    * @return the built vector store
    * @throws IOException Something went wrong finalizing the build
    */
   public abstract VectorStore build() throws IOException;

   /**
    * Gets the dimension of the vectors in the store
    *
    * @return the dimension of the vectors in the story
    */
   public int dimension() {
      return this.dimension;
   }

   /**
    * Sets the measure used for doing nearest neighbor queries
    *
    * @param measure the measure to use
    * @return the vector store builder
    */
   public final VectorStoreBuilder measure(Measure measure) {
      this.measure = measure;
      return this;
   }

   /**
    * Gets the measure used for nearest neighbors
    *
    * @return the measure used for nearest neighbor calculations
    */
   public Measure measure() {
      return this.measure;
   }


}// END OF VectorStoreBuilder
