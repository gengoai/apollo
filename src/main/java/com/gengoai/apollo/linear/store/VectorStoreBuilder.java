package com.gengoai.apollo.linear.store;

import com.gengoai.apollo.linear.NDArray;
import com.gengoai.collection.map.NormalizedStringMap;

import java.io.IOException;
import java.util.Collections;
import java.util.Map;
import java.util.Set;
import java.util.stream.Stream;

/**
 * Abstract base builder for {@link VectorStore}s.
 *
 * @author David B. Bracewell
 */
public abstract class VectorStoreBuilder  {
   public static final String DIMENSION = "DIMENSION";
   private final Map<String, Object> parameters = new NormalizedStringMap<>();

   public VectorStoreBuilder() {
      parameters.put(DIMENSION, -1);
   }

   public VectorStoreBuilder parameter(String name, Object value) {
      parameters.put(name, value);
      return this;
   }

   public <T> T parameterAs(String name, Class<T> clazz) {
      Object value = parameters.get(name);
      return value == null ? null : clazz.cast(value);
   }

   public Map<String, Object> parameterMap() {
      return Collections.unmodifiableMap(parameters);
   }

   public Set<String> parameterNames() {
      return Collections.unmodifiableSet(parameters.keySet());
   }

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
   public VectorStoreBuilder addAll(Iterable<NDArray> vectors) {
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
   public VectorStoreBuilder addAll(Stream<NDArray> vectors){
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
      return parameterAs(DIMENSION, Integer.class);
   }

   public VectorStoreBuilder dimension(int dimension) {
      return parameter(DIMENSION, dimension);
   }


}// END OF VectorStoreBuilder
