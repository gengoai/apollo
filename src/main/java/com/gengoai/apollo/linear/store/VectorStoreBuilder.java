package com.gengoai.apollo.linear.store;

import com.gengoai.apollo.linear.NDArray;
import com.gengoai.apollo.stat.measure.Measure;
import com.gengoai.apollo.stat.measure.Similarity;
import com.gengoai.guava.common.base.Preconditions;
import com.gengoai.mango.conversion.Cast;
import lombok.Getter;
import lombok.NonNull;
import lombok.experimental.Accessors;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

/**
 * The interface Vector store builder.
 *
 * @param <KEY> the type parameter
 * @author David B. Bracewell
 */
@Accessors(fluent = true)
public abstract class VectorStoreBuilder<KEY> {
   /**
    * The Vectors.
    */
   protected final Map<KEY, NDArray> vectors = new HashMap<>();
   @Getter
   private int dimension = 100;
   @Getter
   private Measure measure = Similarity.Cosine;

   /**
    * Add a vector to the vector store.
    *
    * @param vector the vector to add
    * @return this vector store builder
    * @throws NullPointerException if the vector or its key is null
    */
   public final VectorStoreBuilder<KEY> add(@NonNull NDArray vector) {
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
   public final VectorStoreBuilder<KEY> add(@NonNull KEY key, @NonNull NDArray vector) {
      vectors.put(key, vector.copy().setLabel(key));
      return this;
   }

   /**
    * Adds all vectors in the given iterable to the vector store
    *
    * @param vectors the vectors to add
    * @return this vector store builder
    * @throws NullPointerException if the vectors or their keys are null
    */
   public final VectorStoreBuilder<KEY> addAll(@NonNull Iterable<NDArray> vectors) {
      vectors.forEach(this::add);
      return this;
   }

   /**
    * Finalizes the building of the {@link VectorStore}.
    *
    * @return the built vector store
    * @throws IOException Something went wrong finalizing the build
    */
   public abstract VectorStore<KEY> build() throws IOException;

   /**
    * Sets the dimension of the vectors
    *
    * @param dimension the dimension
    * @return the vector store builder
    */
   public final VectorStoreBuilder<KEY> dimension(int dimension) {
      Preconditions.checkArgument(dimension > 0, "Dimension must be > 0");
      this.dimension = dimension;
      return this;
   }

   /**
    * Sets the measure used for doing nearest neighbor queries
    *
    * @param measure the measure to use
    * @return the vector store builder
    */
   public final VectorStoreBuilder<KEY> measure(@NonNull Measure measure) {
      this.measure = measure;
      return this;
   }

   /**
    * Removes a vector from the store given its key.
    *
    * @param key the key of the vector to remove
    * @return the vector associated with the key or null if no vector is assigned to that key
    * @throws NullPointerException if the key is null
    */
   public final NDArray remove(@NonNull KEY key) {
      return vectors.remove(key);
   }

   /**
    * Removes the given vector from the store.
    *
    * @param vector the vector to remove
    * @return True if the vector was removed, False if not
    * @throws NullPointerException if the vector or its label is null
    */
   public final boolean remove(@NonNull NDArray vector) {
      return remove(Cast.<KEY>as(vector.getLabel())) != null;
   }

}// END OF VectorStoreBuilder
