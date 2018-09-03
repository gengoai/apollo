package com.gengoai.apollo.linear.store;

import com.gengoai.Validation;
import com.gengoai.apollo.linear.NDArray;
import com.gengoai.apollo.stat.measure.Measure;
import com.gengoai.apollo.stat.measure.Similarity;
import com.gengoai.conversion.Cast;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

import static com.gengoai.Validation.notNull;

/**
 * Abstract base builder for {@link VectorStore}s.
 *
 * @param <KEY> the key type parameter
 * @author David B. Bracewell
 */
public abstract class VectorStoreBuilder<KEY> {
   protected final Map<KEY, NDArray> vectors = new HashMap<>();
   private int dimension = 100;
   private Measure measure = Similarity.Cosine;


   /**
    * Add a vector to the vector store. The key is expected to be the label of the vector
    *
    * @param vector the vector to add
    * @return this vector store builder
    * @throws NullPointerException if the vector or its key is null
    */
   public final VectorStoreBuilder<KEY> add(NDArray vector) {
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
   public final VectorStoreBuilder<KEY> add(KEY key, NDArray vector) {
      notNull(key, "Key cannot be null");
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
   public final VectorStoreBuilder<KEY> addAll(Iterable<NDArray> vectors) {
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
      Validation.checkArgument(dimension > 0, "Dimension must be > 0");
      this.dimension = dimension;
      return this;
   }

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
   public final VectorStoreBuilder<KEY> measure(Measure measure) {
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

   /**
    * Removes a vector from the store given its key.
    *
    * @param key the key of the vector to remove
    * @return the vector associated with the key or null if no vector is assigned to that key
    * @throws NullPointerException if the key is null
    */
   public final NDArray remove(KEY key) {
      return vectors.remove(key);
   }

   /**
    * Removes the given vector from the store.
    *
    * @param vector the vector to remove
    * @return True if the vector was removed, False if not
    * @throws NullPointerException if the vector or its label is null
    */
   public final boolean remove(NDArray vector) {
      return remove(Cast.<KEY>as(vector.getLabel())) != null;
   }

}// END OF VectorStoreBuilder
