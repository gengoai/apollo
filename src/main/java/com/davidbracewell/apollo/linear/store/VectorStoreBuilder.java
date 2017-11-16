package com.davidbracewell.apollo.linear.store;

import com.davidbracewell.apollo.linear.NDArray;
import com.davidbracewell.apollo.stat.measure.Measure;
import com.davidbracewell.apollo.stat.measure.Similarity;
import com.davidbracewell.conversion.Cast;
import com.davidbracewell.guava.common.base.Preconditions;
import com.davidbracewell.io.Commitable;
import lombok.Getter;
import lombok.NonNull;

import java.io.IOException;

/**
 * The interface Vector store builder.
 *
 * @param <KEY> the type parameter
 * @author David B. Bracewell
 */
public abstract class VectorStoreBuilder<KEY> implements Commitable {
   /**
    * The Dimension.
    */
   @Getter
   private final int dimension;
   /**
    * The Measure.
    */
   @Getter
   private Measure measure = Similarity.Cosine;

   /**
    * Instantiates a new Vector store builder.
    *
    * @param dimension the dimension
    */
   protected VectorStoreBuilder(int dimension) {
      Preconditions.checkArgument(dimension > 0, "Vector dimension must be > 0");
      this.dimension = dimension;
   }

   /**
    * Add a vector to the vector store.
    *
    * @param vector the vector to add
    * @return this vector store builder
    * @throws NullPointerException if the vector or its key is null
    */
   public VectorStoreBuilder<KEY> add(@NonNull NDArray vector) {
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
   public abstract VectorStoreBuilder<KEY> add(KEY key, NDArray vector);

   /**
    * Adds all vectors in the given iterable to the vector store
    *
    * @param vectors the vectors to add
    * @return this vector store builder
    * @throws NullPointerException if the vectors or their keys are null
    */
   public VectorStoreBuilder<KEY> addAll(@NonNull Iterable<NDArray> vectors) {
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
   public abstract NDArray remove(KEY key);

   /**
    * Removes the given vector from the store.
    *
    * @param vector the vector to remove
    * @return True if the vector was removed, False if not
    * @throws NullPointerException if the vector or its label is null
    */
   public boolean remove(@NonNull NDArray vector) {
      return remove(Cast.<KEY>as(vector.getLabel())) != null;
   }

}// END OF VectorStoreBuilder
